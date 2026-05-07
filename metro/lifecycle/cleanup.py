"""Process cleanup — tmux session termination (local and remote)."""

from __future__ import annotations

import subprocess
import time


def kill_local_tmux_sessions(prefix: str = ""):
    """Kill local tmux sessions whose name starts with *prefix*."""
    label = prefix or "all"
    print(f"Killing local tmux sessions ({label})...")

    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  No tmux sessions found.")
        return

    for name in result.stdout.strip().split("\n"):
        if not name:
            continue
        if prefix and not name.startswith(prefix):
            continue
        print(f"  Killing tmux session: {name}")
        subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)


def kill_remote_tmux_sessions(
    prefix: str,
    remote_ips: list[str],
    ssh_port: int = 22,
    ssh_user: str = "root",
):
    """Kill remote tmux sessions matching *prefix* via SSH."""
    print(f"Killing remote tmux sessions ({prefix})...")

    for ip in remote_ips:
        print(f"  Killing on {ip}...")
        cmd = (
            f"ssh -p {ssh_port} {ssh_user}@{ip} "
            f'"for sess in $(tmux list-sessions -F \'#{{session_name}}\' 2>/dev/null | grep \'^{prefix}\'); do '
            f'echo \"  Killing: $sess\"; tmux kill-session -t \"$sess\"; done" '
            f"2>/dev/null"
        )
        subprocess.run(cmd, shell=True, capture_output=True)


def cleanup_cluster(
    venv_path: str = "",
    work_dir: str = "",
    remote_ips: list[str] | None = None,
    remote_venv_path: str | None = None,
    ssh_port: int = 22,
    ssh_user: str = "root",
    local_session_prefix: str = "bailian",
    remote_session_prefix: str = "metric_test_",
):
    """Clean up all experiment processes by killing tmux sessions.

    Local session (e.g. \"bailian\") and remote sessions (e.g. \"metric_test_*\")
    are terminated.  All child processes exit with the session.
    """
    kill_local_tmux_sessions(prefix=local_session_prefix)

    if remote_ips:
        kill_remote_tmux_sessions(
            remote_session_prefix, remote_ips, ssh_port, ssh_user,
        )

    time.sleep(2)
    print("Cleanup complete.")


def cleanup_tmux_session(session_name: str):
    """Kill a tmux session by name."""
    check = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        print(f"Cleaning up tmux session '{session_name}'...")
        subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)
