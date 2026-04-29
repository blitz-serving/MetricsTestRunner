"""Experiment orchestrator — single experiment lifecycle management.

This is the Python replacement for the 700-line ``bailian_dash_12_qwen235b.sh``.
It orchestrates: build → cleanup → launch backends → health wait → launch router
→ health wait → run client → collect → cleanup.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime

from metro.types import ExperimentConfig, ExperimentResult, ClusterProfile
from metro.config.cluster import cluster_to_variables
from metro.runtime.process import ProcessTracker
from metro.lifecycle.build import build_all
from metro.lifecycle.backend import launch_backends
from metro.lifecycle.router import launch_router
from metro.lifecycle.client import run_client
from metro.lifecycle.cleanup import cleanup_cluster, cleanup_tmux_session


class Experiment:
    """Orchestrates a single evaluation experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tracker = ProcessTracker()
        self._start_time: float | None = None
        self._output_dir: str = ""
        self._remote_output_dir: str = ""

    def run(self, dry_run: bool = False) -> ExperimentResult:
        """Run the full experiment lifecycle.

        1. Build binaries (router_v2 + client)
        2. Clean up old processes
        3. Launch vLLM backends and wait for health
        4. Launch router and wait for health
        5. Run client until completion
        6. Clean up processes
        7. Collect results

        If *dry_run* is True, prints the commands that would be run
        without executing them.
        """
        self._start_time = time.time()
        experiment_id = self._generate_id()

        # Set up output directories
        self._setup_output_dirs(experiment_id)
        variables = self._build_variables()

        profile = self.config.cluster

        try:
            # Phase 1: Build
            if not self.config.no_backend:
                print(f"\n{'='*60}")
                print(f"PHASE 1: Building binaries")
                print(f"{'='*60}")
                if not build_all(
                    work_dir=profile.local.work_dir,
                    features=self.config.policy,
                ):
                    return self._fail(experiment_id, "Build failed")

            # Phase 2: Cleanup old processes
            print(f"\n{'='*60}")
            print(f"PHASE 2: Cleaning up old processes")
            print(f"{'='*60}")
            remote_ips = [n.ip for n in profile.remote.nodes] if self.config.use_remote else []
            remote_venv = profile.remote.nodes[0].venv_path if profile.remote.nodes else None
            cleanup_cluster(
                venv_path=profile.local.venv_path,
                work_dir=profile.local.work_dir,
                remote_ips=remote_ips if self.config.use_remote else None,
                remote_venv_path=remote_venv if self.config.use_remote else None,
                ssh_port=profile.remote.ssh_port,
                ssh_user=profile.remote.user,
            )
            cleanup_tmux_session("bailian")

            # Phase 3: Launch backends
            if not self.config.no_backend:
                print(f"\n{'='*60}")
                print(f"PHASE 3: Launching vLLM backends")
                print(f"{'='*60}")
                if not launch_backends(
                    backend_toml=self.config.backend_toml,
                    variables=variables,
                    tracker=self.tracker,
                    use_remote=self.config.use_remote,
                ):
                    return self._fail(experiment_id, "Backend launch failed")

            # Phase 4: Launch router
            print(f"\n{'='*60}")
            print(f"PHASE 4: Launching router")
            print(f"{'='*60}")
            if not launch_router(
                router_toml=self.config.router_toml,
                variables=variables,
                tracker=self.tracker,
            ):
                return self._fail(experiment_id, "Router launch failed")

            # Phase 5: Run client
            print(f"\n{'='*60}")
            print(f"PHASE 5: Running client")
            print(f"{'='*60}")
            client_binary = f"{profile.local.work_dir}/target/release/client"
            run_client(
                client_toml=self.config.client_toml,
                variables=variables,
                tracker=self.tracker,
                time_in_secs=self.config.time_in_secs,
                client_binary_path=client_binary,
            )

            duration = time.time() - self._start_time
            return ExperimentResult(
                experiment_id=experiment_id,
                status="success",
                output_dir=self._output_dir,
                duration_secs=round(duration, 1),
                cluster=profile.name,
                model=self.config.model_name,
                grid_point={"policy": self.config.policy},
            )

        except KeyboardInterrupt:
            print("\nExperiment interrupted by user.")
            return self._fail(experiment_id, "Interrupted")
        except Exception as e:
            print(f"\nExperiment error: {e}")
            import traceback
            traceback.print_exc()
            return self._fail(experiment_id, str(e))
        finally:
            # Phase 6: Cleanup
            print(f"\n{'='*60}")
            print(f"PHASE 6: Cleanup")
            print(f"{'='*60}")
            self.tracker.cleanup_all()
            cleanup_cluster(
                venv_path=profile.local.venv_path,
                work_dir=profile.local.work_dir,
                remote_ips=remote_ips if self.config.use_remote else None,
                remote_venv_path=remote_venv if self.config.use_remote else None,
                ssh_port=profile.remote.ssh_port,
                ssh_user=profile.remote.user,
            )

    def _generate_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_{self.config.policy}"

    def _setup_output_dirs(self, experiment_id: str):
        profile = self.config.cluster

        if self.config.output_dir:
            self._output_dir = self.config.output_dir
        else:
            self._output_dir = os.path.join(profile.local.output_base, experiment_id)

        if self.config.remote_output_dir:
            self._remote_output_dir = self.config.remote_output_dir
        elif profile.remote.nodes:
            self._remote_output_dir = os.path.join(
                profile.remote.nodes[0].output_base, experiment_id
            )

        os.makedirs(self._output_dir, exist_ok=True)

        # Copy config files to output dir for reproducibility
        for src, name in [
            (self.config.backend_toml, "backend.toml"),
            (self.config.router_toml, "router.toml"),
            (self.config.client_toml, "client.toml"),
        ]:
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(self._output_dir, name))

    def _build_variables(self) -> dict:
        profile = self.config.cluster
        variables = cluster_to_variables(
            profile, self.config.model_name, self.config.use_remote,
        )
        # Override output dirs
        variables["output-dir"] = self._output_dir
        if self._remote_output_dir:
            variables["remote-output-dir"] = [self._remote_output_dir]

        # Inject time-in-secs so client TOML's ${time-in-secs} macro resolves
        variables["time-in-secs"] = str(self.config.time_in_secs)

        # Apply user overrides
        variables.update(self.config.overrides)

        return variables

    def _fail(self, experiment_id: str, error: str) -> ExperimentResult:
        duration = time.time() - (self._start_time or time.time())
        return ExperimentResult(
            experiment_id=experiment_id,
            status="failed",
            output_dir=self._output_dir,
            duration_secs=round(duration, 1),
            cluster=self.config.cluster.name,
            model=self.config.model_name,
            error=error,
        )
