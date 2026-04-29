"""Result collector — merge JSONL logs, fetch remote files, sync to NFS."""

from __future__ import annotations

import glob
import os
import subprocess


def merge_client_jsonl(output_dir: str) -> str | None:
    """Merge all client*.jsonl files into a single client.jsonl."""
    pattern = os.path.join(output_dir, "client*.jsonl")
    jsonl_files = sorted(glob.glob(pattern))

    if not jsonl_files:
        print(f"No client JSONL files found in {output_dir}")
        return None

    merged_path = os.path.join(output_dir, "client.jsonl")
    sources = [f for f in jsonl_files if os.path.basename(f) != "client.jsonl"]
    if not sources:
        # Only client.jsonl exists — nothing to merge, do not truncate.
        return merged_path

    with open(merged_path, "w") as out:
        for fpath in sources:
            with open(fpath, "r") as inp:
                out.write(inp.read())

    print(f"Merged {len(sources)} JSONL files into {merged_path}")
    return merged_path


def sync_to_nfs(
    source_dir: str,
    store_base: str,
    remove_source: bool = True,
) -> bool:
    """Rsync experiment results from /tmp to persistent NFS storage.

    Safety: only removes source if it's under /tmp/.
    """
    if not os.path.isdir(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return False

    os.makedirs(store_base, exist_ok=True)

    result = subprocess.run(
        ["rsync", "-av", source_dir, f"{store_base}/"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"rsync failed: {result.stderr}")
        return False

    if remove_source and source_dir.startswith("/tmp/"):
        subprocess.run(["rm", "-rf", source_dir])
        print(f"Removed source {source_dir}")

    return True


def sync_remote_to_nfs(
    remote_ip: str,
    remote_dir: str,
    store_base: str,
    ssh_port: int = 22,
    ssh_user: str = "root",
    remove_source: bool = True,
) -> bool:
    """Rsync remote experiment results to remote NFS storage."""
    cmd = (
        f"ssh -p {ssh_port} {ssh_user}@{remote_ip} \""
        f"if [[ '{remote_dir}' != /tmp/* ]] || [[ ! -d '{remote_dir}' ]]; then "
        f"echo 'ERROR: invalid dir' >&2; exit 1; fi; "
        f"rsync -av '{remote_dir}' '{store_base}/';"
    )
    if remove_source:
        cmd += f" rm -rf '{remote_dir}';"
    cmd += '"'

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Remote sync failed on {remote_ip}: {result.stderr}")
        return False

    return True


def collect_artifacts(output_dir: str) -> dict[str, str | list[str]]:
    """Gather paths to experiment artifacts."""
    artifacts = {}

    client_jsonl = os.path.join(output_dir, "client.jsonl")
    if os.path.isfile(client_jsonl):
        artifacts["client_jsonl"] = client_jsonl

    router_log = os.path.join(output_dir, "router_v2.log")
    if os.path.isfile(router_log):
        artifacts["router_log"] = router_log

    cpu_log = os.path.join(output_dir, "cpu_usage.log")
    if os.path.isfile(cpu_log):
        artifacts["cpu_log"] = cpu_log

    figures = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    if figures:
        artifacts["figures"] = figures

    return artifacts
