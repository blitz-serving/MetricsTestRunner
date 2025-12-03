import argparse
import subprocess
import tomli as tomllib
import time
import os
import signal
import re
from typing import Any, List, Dict, Tuple, Optional, Union
import copy
import json
import sys
from functools import reduce

# buffer background processes for cleanup on exit
background_procs = []

# runtime registry
rt_registry = {}

def register(name):
    def decorator(func):
        rt_registry[name] = func
        return func

    return decorator

def param_expand(s: str, params: dict) -> str:
    def replacer(match):
        var = match.group(1)
        return str(params.get(var, match.group(0)))
    result = re.sub(r'\$\{([^}]+)\}', replacer, s)
    if result != s:
        print(f'"{s}" |-> {result}')
    return result

def resolve_inheritance(app_name: str, app_config: dict, seen: tuple = None) -> dict:
    if seen is None:
        seen = ()
    if app_name in seen:
        raise ValueError(
            f"Circular inheritance detected: {' -> '.join(seen + (app_name,))}"
        )

    app_def = app_config.get(app_name)
    if app_def is None:
        raise ValueError(f"App '{app_name}' not found in config.")

    # create deepcopy 
    result = copy.deepcopy(app_def)

    # process inherit
    if "inherit" in app_def:
        parent_name = app_def["inherit"]
        parent = resolve_inheritance(parent_name, app_config, seen + (app_name,))
        # merge father first
        for k, v in parent.items():
            if k not in result:
                result[k] = copy.deepcopy(v)
            elif k == "config" or k == "macro_rules":
                # merge dict, child first
                merged = copy.deepcopy(v)
                merged.update(result[k])
                result[k] = merged
            elif k == "extra_args":
                if app_def.get(k) is None:
                    result[k] = copy.deepcopy(v)
        result.pop("inherit", None)

    return result


def set_global_variables(variables: dict, global_kv: dict):
    for key, value in variables.items():
        if isinstance(value, str):
            variables[key] = param_expand(value, global_kv)


def load_toml_config(path: str, global_kv: dict = {}):
    with open(path, "rb") as f:
        config = tomllib.load(f)

    # get all variables
    variables = config.get("variables", {})

    # fill and replace all variables
    set_global_variables(variables, global_kv)
    config = resolve_variables(config, variables)

    # process all app's inherit
    resolved_app_config = {}
    app_config_raw = config.get("app", {})

    for app_name, app_def in app_config_raw.items():
        resolved_app_config[app_name] = resolve_inheritance(app_name, app_config_raw)

    config["app"] = resolved_app_config
    return config, variables | global_kv # merge these two dicts


def resolve_variables(obj, variables: dict) -> Any:
    """replace ${key} as the value recursively"""
    if isinstance(obj, str):
        return param_expand(obj, variables)
    elif isinstance(obj, list):
        return [resolve_variables(item, variables) for item in obj]
    elif isinstance(obj, dict):
        return {k: resolve_variables(v, variables) for k, v in obj.items()}
    else:
        return obj


def insert_envs(old_cmd: str, app_running_config: dict) -> str:
    envs = app_running_config.get("envs", {})
    env_str = ""
    for key, value in envs.items():
        env_str += f"{key}={value} "
    return env_str + old_cmd


def process_macro(
    app_name: str, 
    app_general: dict, 
    app_self_cfg: dict,
    add_executable: bool = True
) -> str:
    macro_rules = app_general.get("macro_rules", {})
    args = []

    for cli_key, config_key in macro_rules.items():
        value = app_self_cfg.get(config_key, "")
        if value is None or value == "":
            value = "''"
        args.append(f"{cli_key} {value}")

    executable = app_general["executable"]
    # print(f"exe {executable}")

    if isinstance(executable, str):
        cmd_parts = [executable]
    elif isinstance(executable, list):
        cmd_parts = [" ".join([str(x) for x in executable])]
    else:
        cmd_parts = [str(executable)]

    extra_args = app_general.get("extra_args", [])
    if isinstance(extra_args, str):
        extra_args = [extra_args]

    if add_executable:
        cmd_parts = cmd_parts + extra_args + args
    else:
        cmd_parts = extra_args + args

    # print(f"cmd: {" ".join(cmd_parts)}")

    return " ".join(cmd_parts)


def kill_local_process_on_port(port: int):
    cmd = f"lsof -i :{port} -t | xargs -r -- kill -9"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to kill process on port {port}: {result.stderr.strip()}")
    else:
        print(f"Sent SIGKILL to process(es) on port {port}")

def kill_remote_process_on_port(remote_host: str, remote_user: str, remote_ssh_port: int, port: int):
    """Kill remote process occupying a specific port using kill -9."""
    try:
        ssh_cmd = [
            "ssh",
            "-p", str(remote_ssh_port),
            f"{remote_user}@{remote_host}",
            f"lsof -i :{port} -t | xargs -r kill -9"
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Killed remote process(es) on port {port} on {remote_host}.")
        else:
            if result.stderr:
                print(f"Error killing remote process on port {port} on {remote_host}: {result.stderr.strip()}")
            else:
                print(f"No process found on port {port} on {remote_host}.")
    except Exception as e:
        print(f"Error killing remote process on port {port} on {remote_host}: {e}")

@register("raw")
def gen_raw_cmd(app_cmd: str, 
                rt_config: dict, 
                app_running_config: dict,
                variables: dict) -> str:
    app = app_running_config["app"]
    stubs = []
    stub_path = f"{variables.get('work-dir', '')}/exps/blitz-run/configs/config-stubs.json"
    if stub_path and os.path.exists(stub_path):
        with open(stub_path, "r") as f:
            data = json.load(f)
            stubs = data

    if app == "vllm_template" or app == "vllm_remote_template":
        base_port = variables.get("base_vllm_port", 22222)
        offset = app_running_config.get("port_offset", 0)
        role = app_running_config.get("role", "kv_producer")
        kv_port = app_running_config.get("kv_port", 22281)
        
        # print(f"{base_port}, {offset=}")
        # print(f"{rt_config=} {app_running_config=}")
        # print(f"before replace {app_cmd=}")
        port = base_port + offset
        stubs.append(f"http://localhost:{port}")
        
        # Create output directory if it doesn't exist
        output_dir = variables.get("output_dir", "/tmp")
        os.makedirs(output_dir, exist_ok=True)
        log_file = f"{output_dir}/vllm{offset+1}.log"
        
        # Kill any existing processes on these ports before starting new ones
        kill_local_process_on_port(port)
        kill_local_process_on_port(kv_port)
        
        # replace the variables in command
        app_cmd = app_cmd.replace("${port}", str(port))
        app_cmd = app_cmd.replace("${log_file}", log_file)
        app_cmd = app_cmd.replace("${role}", str(role))
        app_cmd = app_cmd.replace("${kv_port}", str(kv_port))

        print(f"replace stubs to {stubs}")
        if stub_path:
            with open(stub_path, "w") as f:
                json.dump(stubs, f, indent=2)

    # insert some envs
    full_cmd = insert_envs(app_cmd, app_running_config)

    # run background
    if app_running_config.get("background", False):
        return f"{full_cmd} &"
    return full_cmd


@register("ssh")
def gen_ssh_cmd(app_cmd: str, 
                ssh_global_config: dict, 
                app_running_config: dict,
                variables: dict) -> str:
    # Extract SSH index from runtime name (e.g., "ssh_1" -> 1)
    rt_name = ssh_global_config.get("__rt_name", "ssh")
    if "_" in rt_name:
        ssh_index = int(rt_name.split("_")[1])
    else:
        ssh_index = 0

    # Get remote IPs from variables
    remote_ips = variables.get("remote-ips", [])
    if not remote_ips or ssh_index >= len(remote_ips):
        raise ValueError(f"No remote IP found for SSH index {ssh_index} in {remote_ips}")
    
    remote_host = remote_ips[ssh_index]
    remote_user = variables.get("remote_user", "root")
    remote_ssh_port = variables.get("remote_ssh_port", 22)

    # Handle port offset for remote execution
    app = app_running_config["app"]
    stubs = []
    stub_path = f"{variables.get('work-dir', '')}/exps/blitz-run/configs/config-stubs.json"
    
    if stub_path and os.path.exists(stub_path):
        with open(stub_path, "r") as f:
            data = json.load(f)
            stubs = data
    
    if app == "vllm_remote_template":
        # Get remote base port - use different base ports for different machines
        remote_base_ports = variables.get("remote_base_vllm_ports", [])
        if not remote_base_ports or ssh_index >= len(remote_base_ports):
            base_port = 59180 + ssh_index * 100  # Default base port with offset per machine
        else:
            base_port = remote_base_ports[ssh_index]
            
        offset = app_running_config.get("port_offset", 0)
        role = app_running_config.get("role", "kv_producer")
        kv_port = app_running_config.get("kv_port", 22281)
        
        port = base_port + offset
        stubs.append(f"http://{remote_host}:{port}")
        
        # Get remote output directory for this specific machine
        remote_output_dirs = variables.get("remote-output-dir", [])
        if not remote_output_dirs or ssh_index >= len(remote_output_dirs):
            raise ValueError(f"No remote output directory found for SSH index {ssh_index}")
        
        remote_output_dir = remote_output_dirs[ssh_index]
        log_file = f"{remote_output_dir}/vllm{offset+1}.log"
        
        # Kill any existing processes on these ports before starting new ones
        kill_remote_process_on_port(remote_host, remote_user, remote_ssh_port, port)
        kill_remote_process_on_port(remote_host, remote_user, remote_ssh_port, kv_port)
        
        # Replace variables in command
        app_cmd = app_cmd.replace("${port}", str(port))
        app_cmd = app_cmd.replace("${log_file}", log_file)
        app_cmd = app_cmd.replace("${role}", str(role))
        app_cmd = app_cmd.replace("${kv_port}", str(kv_port))
        
        print(f"Adding remote stubs to {stubs}")
        if stub_path:
            with open(stub_path, "w") as f:
                json.dump(stubs, f, indent=2)

    # Insert environment variables (e.g., CUDA_VISIBLE_DEVICES=7 ...)
    app_cmd = insert_envs(app_cmd, app_running_config)

    envs = app_running_config.get("envs", {})
    #print(f"remote {envs=} {app_running_config=} {app_cmd=}")
    cuda_visible = envs.get("CUDA_VISIBLE_DEVICES", "0")  # default to "0" if not set
    # Handle cases like "7" or "7,8" — take first device
    gpu_id = str(cuda_visible).split(",")[0].strip()

    tmux_session_name = f"metric_test_{gpu_id}"

    # Escape quotes and backslashes for safe shell embedding
    escaped_app_cmd = app_cmd.replace("'", "'\"'\"'")  # safely escape single quotes

    # Construct the remote shell command that:
    # 1. Kills existing tmux session (if any)
    # 2. Starts a new detached tmux session running the app_cmd with proper redirection
    remote_shell_cmd = (
        f"tmux kill-session -t {tmux_session_name} 2>/dev/null || true; "
        f"mkdir -p {remote_output_dir} && "  # Ensure directory exists
        f"tmux new-session -d -s {tmux_session_name} '"
        f"{escaped_app_cmd}; exec bash"
        f"'"
    )

    # Wrap in ssh call
    full_cmd = f'ssh -p {remote_ssh_port} "{remote_user}@{remote_host}" "{remote_shell_cmd}"'

    return full_cmd


@register("mpi")
def gen_mpi_cmd(app_cmd: str, mpi_global_config: dict, app_running_config: dict) -> str:
    hosts = app_running_config.get("hosts", {})
    host_list = [f"{k}:{v}" for k, v in hosts.items()]
    host_cmd = f"--host {','.join(host_list)}" if host_list else ""

    app_cmd = insert_envs(app_cmd, app_running_config)
    mpirun_cmd = f"mpirun {host_cmd}".strip() if host_cmd else "mpirun"
    full_cmd = f"{mpirun_cmd} {app_cmd}"
    return full_cmd


def block_until_keyword(proc: subprocess.Popen, keyword: str):
    try:
        while True:
            line = proc.stdout.readline().decode("utf-8")
            if not line:
                break
            if keyword in line:
                print(f"Found keyword: '{keyword}'")
                break
    except:
        pass


import ast

def select_app_config(app_general, index_id):
    # Step 1: 处理 executable
    exec_single_str = app_general['executable'][0]  # "['a','b','c']/bin/vllm serve"
    
    # 找到第一个 ']' 的位置（因为列表结束后立即是 /bin/...）
    try:
        bracket_end = exec_single_str.index(']') + 1
    except ValueError:
        raise ValueError("Invalid format in executable: missing ']'")
    
    list_part = exec_single_str[:bracket_end]      # "['a','b','c']"
    suffix = exec_single_str[bracket_end:]         # "/bin/vllm serve"
    
    base_exec_paths = ast.literal_eval(list_part)
    
    if not (0 <= index_id < len(base_exec_paths)):
        raise IndexError(f"index_id {index_id} out of range for {len(base_exec_paths)} nodes")
    
    selected_exec = base_exec_paths[index_id] + suffix  # e.g. '/mnt/.../.venvflashinfer/bin/vllm serve'

    # Step 2: 处理 extra_args 中的模型路径（第一个元素是模型路径列表字符串）
    model_list_str = app_general['extra_args'][0]
    model_paths = ast.literal_eval(model_list_str)
    selected_model = model_paths[index_id]

    # Step 3: 合并其余参数
    other_args = app_general['extra_args'][1:]

    res = {
        "executable": [selected_exec],
        "extra_args": [selected_model, *other_args]
    }

    return res


def run_apps(rt: str, rt_config: dict, app_config: dict, variables: dict):
    # Store the original runtime name for SSH indexing
    rt_config_with_name = rt_config.copy()
    rt_config_with_name["__rt_name"] = rt
    
    all_apps = rt_config.get("config", [])
    for init_config in all_apps:
        app_name = init_config["app"]
        if app_name not in app_config:
            raise ValueError(f"App '{app_name}' not found.")

        app_general = app_config[app_name]
        
        app_self_cfg = app_general.get("config", {})
        
        # Create directories based on runtime type
        if rt.split("_")[0] == "raw":
            # For raw runtime, create directories locally
            output_dir = variables.get("output_dir")
            if output_dir:
                print(f"Creating local directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)

            print(f"[before process macro] {app_general=} {app_self_cfg=}")
            app_cmd = process_macro(app_name, app_general, app_self_cfg)
            print(f"[after process macro] {app_cmd=}\n")

        else:
            # For other runtimes (e.g., ssh), create directories remotely
            remote_output_dirs = variables.get("remote-output-dir", [])

            # Extract SSH index from runtime name (e.g., "ssh_1" -> 1)
            if "_" in rt:
                ssh_index = int(rt.split("_")[1])
            else:
                ssh_index = 0
            
            if ssh_index >= len(remote_output_dirs):
                raise ValueError(f"No remote output directory found for SSH index {ssh_index}")
            
            remote_output_dir = remote_output_dirs[ssh_index]
            
            # Get remote IPs from variables
            remote_ips = variables.get("remote_ips", [])
            if not remote_ips or ssh_index >= len(remote_ips):
                raise ValueError(f"No remote IP found for SSH index {ssh_index}")
            
            remote_host = remote_ips[ssh_index]
            remote_user = variables.get("remote_user", "root")
            remote_ssh_port = variables.get("remote_ssh_port", 22)

            # Create remote directory using SSH
            print(f"Creating remote directory: {remote_output_dir} on {remote_host}")
            ssh_cmd = f'ssh -p {remote_ssh_port} "{remote_user}@{remote_host}" "mkdir -p {remote_output_dir}"'
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Failed to create remote directory on {remote_host}: {result.stderr.strip()}")
                # Try to continue anyway
        
            print(f"[before process macro] {app_general=} {app_self_cfg=}")
            app_general = select_app_config(app_general, ssh_index)
            print(f"[after selecting] {app_general=}")
            app_cmd = process_macro(app_name, app_general, app_self_cfg)
            print(f"[after process macro] {app_cmd=}\n")

        rt_func = rt_registry.get(rt.split("_")[0])  # Get base runtime type (e.g., "ssh" from "ssh_1")
        if not rt_func:
            raise ValueError(f"Unknown runtime: {rt}")

        cmd = rt_func(app_cmd, rt_config_with_name, init_config, variables)

        print(f"RUN: {cmd}\n\n")

        if True: # Run
            run_in_bg = init_config.get("background", False)
            keyword = init_config.get("block_keyword")

            if run_in_bg and keyword:
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
                block_until_keyword(proc, keyword)
                background_procs.append(proc)
            elif run_in_bg:
                proc = subprocess.Popen(
                    cmd, shell=True, stderr=subprocess.DEVNULL, preexec_fn=os.setsid
                )
                background_procs.append(proc)
            else:
                subprocess.run(cmd, shell=True, check=True)

        time.sleep(2)


def check_and_get_runtime(config: dict):
    for rt, cfg in config.get("runtime", {}).items():
        yield rt, cfg


def parse_list_argument(arg_value: str) -> List[str]:
    """Parse a string argument that might contain space-separated values or be a single value."""
    if not arg_value:
        return []
    # Split by spaces but preserve quoted values
    return [x.strip() for x in arg_value.split() if x.strip()]


def parse_comma_separated_list(arg_value: str) -> List[str]:
    """Parse a comma-separated string into a list."""
    if not arg_value:
        return []
    return [x.strip() for x in arg_value.split(',') if x.strip()]


def main():

    print("Raw sys.argv:", sys.argv)
    
    parser = argparse.ArgumentParser(description="Run apps from TOML config")
    parser.add_argument("--toml", required=True, help="Path to TOML file")
    # use global kv to replace placeholder..
    args, unknown = parser.parse_known_args()

    # parse global kv given by parameters
    global_kv = {}
    for arg in unknown:
        if arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            # Handle list arguments
            if key in ["remote-ips", "remote-output-dir", "remote-model-path", "remote-venv-path"]:
                # Split by spaces for these specific keys
                global_kv[key] = parse_list_argument(value)
            else:
                global_kv[key] = value

    print(f"{global_kv=}")

    global background_procs
    background_procs = []

    try:
        config, variables = load_toml_config(args.toml, global_kv)
        # print(f"after load and parse vars, {config=}, {variables=} \n")
        # clear stubs file when starting vllm
        work_dir = variables.get("work-dir", "")
        if work_dir and ("vllm_template" in config["app"] or "vllm_remote_template" in config["app"]):
            print(f"clear old stubs")
            stub_path = f"{work_dir}/exps/blitz-run/configs/config-stubs.json"
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "w") as f:
                json.dump([], f, indent=2)
            
        # smart runner need one foreground app as blocker across all runtimes
        has_foreground_app = False
        all_configs = []
        for rt_name, rt_config in check_and_get_runtime(config):
            all_configs.extend(rt_config.get("config", []))
        
        # Check if there's at least one foreground app across all runtimes
        for cfg in all_configs:
            if not cfg.get("background", False):
                has_foreground_app = True
                break
                
        if not has_foreground_app:
            raise ValueError(f"No foreground app found in any runtime config! {config=}")
            
        # Run all apps in all runtimes
        for rt_name, rt_config in check_and_get_runtime(config):
            # print(f"Config app: {config["app"]}")
            run_apps(rt=rt_name, rt_config=rt_config, app_config=config["app"], variables=variables)
            
    except ValueError as e:
        print(f"Fail to run app: {e}")
    except KeyboardInterrupt as e:
        print(f"Keyboard interrupt: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for proc in background_procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=3)
                print(f"Stopped PID={proc.pid}")
            except:
                pass


if __name__ == "__main__":
    main()