"""Process cleanup — local and remote process termination."""

from __future__ import annotations

import subprocess
import time


def kill_processes_by_pattern(pattern: str, description: str = ""):
    """Kill local processes matching *pattern* (SIGTERM then SIGKILL)."""
    label = description or pattern
    print(f"Killing {label} processes (gracefully first)...")

    result = subprocess.run(
        ["pkill", "-f", pattern], capture_output=True, text=True,
    )
    if result.returncode == 0:
        time.sleep(5)
        alive = subprocess.run(
            ["pkill", "-0", "-f", pattern], capture_output=True, text=True,
        )
        if alive.returncode == 0:
            print(f"Some {label} processes still running; forcing kill...")
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def kill_remote_processes_by_pattern(
    pattern: str,
    remote_ips: list[str],
    ssh_port: int = 22,
    ssh_user: str = "root",
    description: str = "",
):
    """Kill processes matching *pattern* on remote machines via SSH."""
    label = description or pattern
    print(f"Killing {label} on remote machines...")

    for ip in remote_ips:
        print(f"  Killing on {ip}...")
        cmd = (
            f"ssh -p {ssh_port} {ssh_user}@{ip} "
            f"\"pkill -f '{pattern}' 2>/dev/null; "
            f"sleep 5; pkill -9 -f '{pattern}' 2>/dev/null\" "
            f"2>/dev/null"
        )
        subprocess.run(cmd, shell=True, capture_output=True)


def cleanup_cluster(
    venv_path: str,
    work_dir: str,
    remote_ips: list[str] | None = None,
    remote_venv_path: str | None = None,
    ssh_port: int = 22,
    ssh_user: str = "root",
):
    """Clean up all experiment processes on local and remote nodes.

    This is the Python equivalent of ``cleanup_processes()`` from
    ``bailian_dash_12_qwen235b.sh``.
    """
    kill_processes_by_pattern(f"{venv_path}/bin/vllm", "vLLM")
    kill_processes_by_pattern("multiprocessing.spawn", "vLLM workers")
    kill_processes_by_pattern("multiprocessing.resource_tracker", "vLLM resource trackers")
    kill_processes_by_pattern(f"{venv_path}/bin/python3 -s", "dead Python")
    kill_processes_by_pattern("target/release/router", "router")
    kill_processes_by_pattern("smart_runner.py", "smart runner")
    kill_processes_by_pattern(f"{work_dir}/target/release/client", "client")
    kill_processes_by_pattern("cpu_monitor.py", "CPU monitor")

    if remote_ips and remote_venv_path:
        kill_remote_processes_by_pattern(
            f"{remote_venv_path}/bin/vllm",
            remote_ips, ssh_port, ssh_user, "remote vLLM",
        )
        kill_remote_processes_by_pattern(
            f"{remote_venv_path}/bin/python3 -s",
            remote_ips, ssh_port, ssh_user, "remote dead Python",
        )

    time.sleep(5)
    print("Cleanup complete.")


def cleanup_tmux_session(session_name: str):
    """Kill a tmux session by name."""
    check = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        print(f"Cleaning up tmux session '{session_name}'...")
        # Send C-c to all panes
        result = subprocess.run(
            ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_id}"],
            capture_output=True, text=True,
        )
        for pane_id in result.stdout.strip().split("\n"):
            if pane_id:
                subprocess.run(["tmux", "send-keys", "-t", pane_id, "C-c"])
        time.sleep(2)
        subprocess.run(["tmux", "kill-session", "-t", session_name])
