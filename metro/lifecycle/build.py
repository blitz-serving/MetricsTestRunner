"""Build manager — compile Rust binaries (router, client)."""

from __future__ import annotations

import os
import shutil
import subprocess


def build_router(
    work_dir: str,
    features: str = "",
    release: bool = True,
    verbose: bool = False,
) -> bool:
    """Build router with optional feature flags (the policy name).

    The router project produces a single ``target/release/router`` binary.
    Switching ``--features`` triggers a full recompile, which is expensive.
    To make sweeps cheap, we cache per-feature binaries as
    ``target/release/router_<features>`` and reuse them when available.
    """
    target_dir = "release" if release else "debug"
    canonical = os.path.join(work_dir, "target", target_dir, "router")
    cached = (
        os.path.join(work_dir, "target", target_dir, f"router_{features}")
        if features else canonical
    )

    if features and os.path.isfile(cached):
        print(f"Using cached router binary: {cached}")
        shutil.copy2(cached, canonical)
        return True

    print(f"Building router with features: {features or '(none)'}")

    cmd = ["cargo", "build", "-p", "router"]
    if release:
        cmd.append("--release")
    if features:
        cmd.extend(["--features", features])
    if not verbose:
        cmd.append("--quiet")

    env_cmd = f"RUSTFLAGS='-Awarnings' {' '.join(cmd)}"
    result = subprocess.run(
        env_cmd, shell=True, cwd=work_dir,
        capture_output=not verbose, text=True,
    )
    if result.returncode != 0:
        print(f"Error: Failed to build router.")
        if result.stderr:
            print(result.stderr[-500:])
        return False

    if features and os.path.isfile(canonical):
        shutil.copy2(canonical, cached)
        print(f"Cached router binary: {cached}")

    print("router build successful.")
    return True


def build_client(
    work_dir: str,
    release: bool = True,
    verbose: bool = False,
) -> bool:
    """Build the request simulator client."""
    print("Building request-sim client...")

    cmd = ["cargo", "build", "-p", "request-sim", "--bin", "request-sim", "-j64"]
    if release:
        cmd.append("--release")
    if not verbose:
        cmd.append("--quiet")

    env_cmd = f"RUSTFLAGS='-Awarnings' {' '.join(cmd)}"
    result = subprocess.run(
        env_cmd, shell=True, cwd=work_dir,
        capture_output=not verbose, text=True,
    )
    if result.returncode != 0:
        print(f"Error: Failed to build client.")
        if result.stderr:
            print(result.stderr[-500:])
        return False

    print("Client build successful.")
    return True


def build_all(
    work_dir: str,
    features: str = "",
    release: bool = True,
    verbose: bool = False,
) -> bool:
    """Build the router. Client is expected to be pre-built (request-sim
    lives in a sibling crate; build it externally)."""
    return build_router(work_dir, features, release, verbose)
