#!/bin/bash

# =============================================================================
# Metrics Test Runner - Bailian Evaluation Script (Multi-Node Version)
# =============================================================================
# This script orchestrates distributed LLM inference experiments using:
# 1. vLLM backends (model serving)
# 2. Router (request routing)
# 3. Client (request generation)
#
# The script creates a tmux session with three windows for each component,
# runs the experiment for a specified duration, collects logs, and generates
# performance visualization figures.
#
# MODIFICATIONS FOR MULTI-NODE SUPPORT:
# - --remote-output-dir accepts a list of directories (one per remote machine)
# - --remote-ips accepts a list of remote machine IPs
# - All remote paths (venv, model, output) are now arrays
# - Process management, startup waiting, and cleanup are distributed across nodes
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration Section - User Configurable Parameters
# -----------------------------------------------------------------------------

# Model path - directory containing the LLM model files
# This is for all 4 machines
MODEL_PATH='/home/admin/resource/model/464482ce.Qwen2.5-7B-Instruct/1.0/'
#REMOTE_MODEL_PATHS will be derived from REMOTE_IPS

# Python virtual environment path with vLLM installed
VENV_PATH='/mnt/debugger/hjb/node1/yaullm/.venvflashinfer'
#REMOTE_VENV_PATHS will be derived from REMOTE_IPS

# Project directory containing the blitz-infer-pack codebase
WORK_DIR='/mnt/debugger/hjb/node1/blitz-infer-pack'

# Flag to skip launching backend (useful for debugging)
NO_BACKEND=false

# Base directory for output logs
OUTPUT_BASE="/tmp/node1/lmmetric-logs"
STORE_OUTPUT_BASE="/mnt/debugger/hjb/node1/lmmetric-logs"

# Directory containing dataset files for client requests
DATASET_DIR="/mnt/debugger/hjb/node1/qwen-bailian-usagetraces-anon"

# Default remote configuration - will be overridden by command line args
REMOTE_IPS=("172.27.21.64")  # Default single node
USE_REMOTE=True
SSH_PORT=10022

# Derived remote paths (will be set based on REMOTE_IPS)
REMOTE_MODEL_PATHS=()
REMOTE_VENV_PATHS=()
REMOTE_OUTPUT_BASES=()
STORE_REMOTE_OUTPUT_BASES=()

# Evaluation duration in seconds
TIME_IN_SEC=$((1200 + 10))

# Session name for tmux
SESSION_NAME="bailian"

# -----------------------------------------------------------------------------
# Derived Configuration - Computed from User Parameters
# -----------------------------------------------------------------------------

# Timestamped output directory for this run
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# These will be overridden if --output-dir or --remote-output-dir are provided
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
REMOTE_OUTPUT_DIRS=()  # Will be set based on REMOTE_IPS or command line args

# -----------------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------------

# Print usage information
print_usage() {
    echo "Usage: $0 [--no-backend] [--work-dir DIR] [--venv-path PATH] [--remote-output-dir DIR1 DIR2...] [--remote-ips IP1 IP2...] [-v] <backend-cfg> <router-cfg> <client-cfg> <policy>"
    echo ""
    echo "Options:"
    echo "  --no-backend          Skip launching vLLM backend processes"
    echo "  --work-dir DIR        Set working directory (default: $WORK_DIR)"
    echo "  --venv-path PATH      Set virtual environment path (default: $VENV_PATH)"
    echo "  --output-dir DIR      Set local output directory (overrides default timestamped path)"
    echo "  --remote-output-dir DIR1 DIR2...   Set remote output directories (one per remote machine)"
    echo "  --remote-ips IP1 IP2...            Set remote machine IPs (default: ${REMOTE_IPS[@]})"
    echo "  -v                    Verbose mode - show build output (default: silent)"
}

# Validate that a file exists
validate_file_exists() {
    local file_path=$1
    local description=$2
    if [ ! -f "$file_path" ]; then
        echo "Error: $description file '$file_path' not found."
        exit 1
    fi
}

# Validate that a directory exists
validate_directory_exists() {
    local dir_path=$1
    local description=$2
    if [ ! -d "$dir_path" ]; then
        echo "Error: $description directory '$dir_path' not found."
        exit 1
    fi
}

# Set up remote paths based on REMOTE_IPS
setup_remote_paths() {
    local num_remotes=${#REMOTE_IPS[@]}
    
    # Clear existing arrays
    REMOTE_MODEL_PATHS=()
    REMOTE_VENV_PATHS=()
    REMOTE_OUTPUT_BASES=()
    STORE_REMOTE_OUTPUT_BASES=()
    
    for ((i=0; i<num_remotes; i++)); do
        remote_ip="${REMOTE_IPS[$i]}"
        
        # Derive paths based on IP or use defaults
        case "$remote_ip" in
            "172.27.21.162")
                REMOTE_MODEL_PATHS+=("${MODEL_PATH}")
                REMOTE_VENV_PATHS+=("/mnt/debugger/hjb/node2/yaullm/.venvflashinfer")
                REMOTE_OUTPUT_BASES+=("/tmp/node2/lmmetric-logs")
                STORE_REMOTE_OUTPUT_BASES+=("/mnt/debugger/hjb/node2/lmmetric-logs")
                ;;
            "172.27.21.163")
                REMOTE_MODEL_PATHS+=("${MODEL_PATH}")
                REMOTE_VENV_PATHS+=("/mnt/debugger/hjb/node3/yaullm/.venvflashinfer")
                REMOTE_OUTPUT_BASES+=("/tmp/node3/lmmetric-logs")
                STORE_REMOTE_OUTPUT_BASES+=("/mnt/debugger/hjb/node3/lmmetric-logs")
                ;;
            "172.27.21.155")
                REMOTE_MODEL_PATHS+=("${MODEL_PATH}")
                REMOTE_VENV_PATHS+=("/mnt/debugger/hjb/node4/yaullm/.venvflashinfer")
                REMOTE_OUTPUT_BASES+=("/tmp/node4/lmmetric-logs")
                STORE_REMOTE_OUTPUT_BASES+=("/mnt/debugger/hjb/node4/lmmetric-logs")
                ;;
            *)
                echo "⚠️  FATAL: unknown remote IP: $remote_ip"
                exit 1
                ;;
        esac
    done
    
    echo "✅ Remote paths configured for ${#REMOTE_IPS[@]} machines:"
    for i in "${!REMOTE_IPS[@]}"; do
        echo "   - ${REMOTE_IPS[$i]}:"
        echo "     Model: ${REMOTE_MODEL_PATHS[$i]}"
        echo "     Venv: ${REMOTE_VENV_PATHS[$i]}"
        echo "     Output base: ${REMOTE_OUTPUT_BASES[$i]}"
        echo "     Storage: ${STORE_REMOTE_OUTPUT_BASES[$i]}"
    done
}

kill_processes_by_pattern() {
    local pattern=$1
    local description=$2
    echo "Killing $description processes (gracefully first)..."
    
    if pkill -f "$pattern" 2>/dev/null; then
        sleep 5
        if pkill -0 -f "$pattern" 2>/dev/null; then
            echo "Some processes still running; forcing kill..."
            pkill -9 -f "$pattern" 2>/dev/null || true
        fi
    fi
}

# Kill processes by pattern on remote machines
kill_remote_processes_by_pattern() {
    local pattern=$1
    local description=$2

    echo "Killing $description processes on remote machines..."

    # Kill processes on each remote machine
    for i in "${!REMOTE_IPS[@]}"; do
        remote_ip="${REMOTE_IPS[$i]}"
        remote_venv_path="${REMOTE_VENV_PATHS[$i]}"
        
        echo "Killing processes on $remote_ip..."

        ssh "-p${SSH_PORT}" "$remote_ip" "
            # source '$remote_venv_path/bin/activate' 2>/dev/null || true
            # Stage 1: Send SIGTERM (default signal)
            if pkill -f '$pattern' 2>/dev/null; then
                sleep 5
                # Stage 2: Check if any matching processes are still alive
                if pkill -0 -f '$pattern' 2>/dev/null; then
                    echo 'Some processes still running; forcing kill...'
                    pkill -9 -f '$pattern' 2>/dev/null || true
                fi
            fi
        " 2>/dev/null || true
    done
}

# Clean up tmux session
cleanup_tmux_session() {
    local session_name=$1
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Cleaning up existing '$session_name' tmux session..."
        
        # Send interrupt signal to all panes
        for pane in $(tmux list-panes -t "$session_name" -F '#{pane_id}'); do
            tmux send-keys -t "$pane" C-c
        done
        
        # Wait for processes to terminate
        sleep 2
        
        # Kill the session
        tmux kill-session -t "$session_name"
    else
        echo "No existing '$session_name' tmux session found."
    fi
}

# Build project components
build_project_components() {
    local features=$1
    local work_dir=$2
    
    echo "Building router_v2 with features: $features"
    
    # Build router_v2 with specified features
    if [ "$VERBOSE" = true ]; then
        RUSTFLAGS="-Awarnings" cargo build -p router_v2  --release --features  "$features"
    else
        RUSTFLAGS="-Awarnings" cargo build -p router_v2  --release --features "$features" --quiet
    fi
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build router_v2."
        exit 1
    fi
    
    # Build request simulator client
    if [ "$VERBOSE" = true ]; then
        RUSTFLAGS="-Awarnings" cargo build -p request-sim --release --bin client -j64
    else
        RUSTFLAGS="-Awarnings" cargo build -p request-sim --release --bin client -j64 --quiet
    fi
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build request-sim client."
        exit 1
    fi
    
    echo "Build successful."
}

# Create output directory and copy config files
setup_output_directory() {
    local output_dir=$1
    local config1=$2
    local config2=$3
    local config3=$4
    local features=$5
    local policy=$6
    local queue="$7/router_v2/src/queue.rs"

    # Create output directory
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"
    
    # Copy configuration files to output directory
    cp "$config1" "$output_dir/backend.toml"
    cp "$config2" "$output_dir/router.toml"
    cp "$config3" "$output_dir/client.toml"
    cp "$queue" "$output_dir/queue.rs"
    
    # Save features string to commands.txt
    echo "$features" > "$output_dir/commands.txt"
    
    echo "$output_dir"
}

# Setup remote output directories
setup_remote_output_directories() {
    local timestamp_dir="$1"
    local policy="$2"
    
    echo "Setting up remote output directories..."
    
    for i in "${!REMOTE_IPS[@]}"; do
        remote_ip="${REMOTE_IPS[$i]}"
        remote_output_base="${REMOTE_OUTPUT_BASES[$i]}"
        remote_output_dir="${remote_output_base}/${timestamp_dir}_${policy}"
        
        echo "  - Creating $remote_output_dir on $remote_ip"
        
        ssh "-p${SSH_PORT}" "$remote_ip" "
            mkdir -p '$remote_output_dir'
            echo 'Remote output directory created successfully'
        "
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to create remote output directory on $remote_ip" >&2
            exit 1
        fi
        
    done
    
    echo "✅ Remote output directories set up successfully"
}

wait_for_vllm_startup() {
    echo "Start to Waiting local vLLM..."
    local remote_dir="$OUTPUT_DIR"
    local max_wait_sec=600 # 7 mins
    local elapsed=0
    local check_interval=5

    if [ -z "$remote_dir" ]; then
        echo "ERROR: OUTPUT_DIR is not set." >&2
        return 1
    fi

    sleep 30
    elapsed=30

    while [ $elapsed -lt $max_wait_sec ]; do
        local all_ready=true

        set -- "$remote_dir"/vllm*.log
        if [ ! -e "$1" ]; then
            echo "No vllm*.log files found in $remote_dir. Waiting..."
            all_ready=false
        else
            for logfile in "$remote_dir"/vllm*.log; do
                if [ ! -f "$logfile" ]; then
                    continue
                fi

                # Check for RuntimeError
                if grep -q "^RuntimeError:" "$logfile" 2>/dev/null; then
                    echo "ERROR: RuntimeError detected in $logfile." >&2
                    return 1
                fi

                # Check for RuntimeError
                if grep -q "^OSError:" "$logfile" 2>/dev/null; then
                    echo "ERROR: OSError: detected in $logfile." >&2
                    return 1
                fi

                expected="INFO:     Application startup complete."
                if ! grep -Fq "$expected" "$logfile" 2>/dev/null; then
                    echo "Waiting for $logfile to complete startup..."
                    all_ready=false
                    break
                fi
            done
        fi

        if [ "$all_ready" = true ]; then
            echo "All vllm*.log files indicate startup complete."
            return 0
        fi

        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    echo "ERROR: vLLM startup timeout after $max_wait_sec seconds." >&2
    return 1
}

wait_for_remote_vllm_startup() {
    echo "Start to Waiting remote vLLM..."
    local max_wait_sec=100
    local elapsed=0
    local check_interval=5

    if [ -z "$REMOTE_OUTPUT_DIRS" ]; then
        echo "ERROR: REMOTE_OUTPUT_DIRS is not set." >&2
        return 1
    fi

    if [ -z "$REMOTE_IPS" ]; then
        echo "ERROR: REMOTE_IPS is not set." >&2
        return 1
    fi

    IFS=',' read -ra IPS <<< "$REMOTE_IPS"
    IFS=',' read -ra DIRS <<< "$REMOTE_OUTPUT_DIRS"

    if [ "${#IPS[@]}" -ne "${#DIRS[@]}" ]; then
        echo "ERROR: Number of REMOTE_IPS (${#IPS[@]}) does not match REMOTE_OUTPUT_DIRS (${#DIRS[@]})." >&2
        return 1
    fi

    elapsed=0
    while [ $elapsed -lt $max_wait_sec ]; do
        local all_ready=true

        for idx in "${!IPS[@]}"; do
            local ip="${IPS[$idx]}"
            local remote_dir="${DIRS[$idx]}"

            # Check if remote log files exist in this directory on this IP
            local remote_log_check
            remote_log_check=$(ssh "-p ${SSH_PORT}" "$ip" "ls ${remote_dir}/vllm*.log 2>/dev/null" 2>/dev/null)

            if [ -z "$remote_log_check" ]; then
                echo "No vllm*.log files found in $remote_dir on $ip. Waiting..."
                all_ready=false
                break
            fi

            # Check each log file on the remote machine
            for remote_logfile in $remote_log_check; do
                # Check for RuntimeError
                if ssh "-p ${SSH_PORT}" "$ip" "grep -q '^RuntimeError:' '$remote_logfile'" 2>/dev/null; then
                    echo "ERROR: RuntimeError detected in $remote_logfile on $ip." >&2
                    return 1
                fi

                # Check for OSError: [Errno 98] (Address already in use, etc.)
                if ssh "-p ${SSH_PORT}" "$ip" "grep -q '^OSError: ' '$remote_logfile'" 2>/dev/null; then
                    echo "ERROR: OSError detected in $remote_logfile on $ip." >&2
                    return 1
                fi

                local expected="INFO:     Application startup complete."
                if ! ssh "-p ${SSH_PORT}" "$ip" "grep -F '$expected' '$remote_logfile'" 2>/dev/null; then
                    echo "Waiting for $remote_logfile on $ip to complete startup..."
                    all_ready=false
                    break 2  # Break out of both the logfile loop and the IP loop
                fi
            done
        done

        if [ "$all_ready" = true ]; then
            echo "All remote vLLM instances indicate startup complete."
            return 0
        fi

        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    echo "ERROR: Remote vLLM startup timeout after $max_wait_sec seconds." >&2
    return 1
}

# Launch tmux session with experiment components
launch_experiment_session() {
    local session_name=$1
    local work_dir=$2
    local venv_path=$3
    local config1=$4
    local config2=$5
    local config3=$6
    local output_base=$7
    local output_dir=$8
    local model_path=$9
    local dataset_dir=${10}
    local no_backend=${11}
    local time_in_sec=${12}
    
    # Source Python virtual environment activation command
    local tmux_cmd="source $venv_path/bin/activate"
    
    # Create new tmux session
    tmux new-session -d -s "$session_name"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Change to that directory
    cd "$SCRIPT_DIR" || exit 1

    # ==========================
    # START CPU MONITOR
    # ==========================
    pkill -9 -f "cpu_monitor.py"
    sleep 3
    local cpu_monitor_log="$output_dir/cpu_usage.log"
    mkdir -p "$output_dir"
    "$venv_path/bin/python" "$SCRIPT_DIR/cpu_monitor.py" -o "$cpu_monitor_log" &
    local cpu_monitor_pid=$!
    echo "📊 Started CPU monitor (PID: $cpu_monitor_pid), logging to $cpu_monitor_log"

    # Launch vLLM backends if not skipped
    if [ "$no_backend" = false ]; then
        echo "🚀 Launching vLLM backends and waiting ..."
        tmux new-window -t "$session_name" -n window1
        tmux send-keys -t "$session_name:window1" "$tmux_cmd && python ../../smart_runner_multiworker.py --toml $config1 --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --remote-output-dir=\"${REMOTE_OUTPUT_DIRS[*]}\" --remote-model-path=\"${REMOTE_MODEL_PATHS[*]}\" --remote-venv-path=\"${REMOTE_VENV_PATHS[*]}\" --remote-ips=\"${REMOTE_IPS[*]}\" --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m        
        # Wait for local vLLM startup with timeout
        if ! wait_for_vllm_startup; then
            echo "❌ FATAL: Local vLLM failed to start in time. Aborting experiment." >&2
            cleanup_processes
            exit 1
        fi
        
        # Wait for remote vLLM startup if we have remote machines
        if [[ "$USE_REMOTE" == "True" ]] && [ ${#REMOTE_IPS[@]} -gt 0 ]; then
            if ! wait_for_remote_vllm_startup; then
                echo "❌ FATAL: Remote vLLM failed to start in time. Aborting experiment." >&2
                cleanup_processes
                exit 1
            fi
        fi
    fi
    
    # Launch router
    echo "atedRoute launching router and waiting 20s..."
    tmux new-window -t "$session_name" -n window2
    tmux send-keys -t "$session_name:window2" "$tmux_cmd && python ../../smart_runner.py --toml $config2 --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
    sleep 20

    # Check if router_v2 process is running
    if ! pgrep -f "router_v2" > /dev/null; then
        echo "❌ FATAL: router_v2 failed to start in time. Aborting experiment." >&2
        cleanup_processes
        exit 1
    fi

    # Launch client
    echo "👥 Launching client..."
    tmux new-window -t "$session_name" -n window3
    tmux send-keys -t "$session_name:window3" "$tmux_cmd && python ../../smart_runner.py --toml $config3 --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
    
    # Wait for experiment to complete
    echo "⏱️  Running experiment for ${time_in_sec}s..."
    sleep $(($time_in_sec))

    client_path="$WORK_DIR/target/release/client"
    while true; do
        if pgrep -f "$client_path" > /dev/null; then
            echo "⏳ Detecting unfinished clients, waiting extra times!!!"
            for i in {1..6}; do
                sleep 5
                if ! pgrep -f "$client_path" > /dev/null; then
                    break
                fi
            done
        else
            break
        fi
    done

    # ==========================
    # STOP CPU MONITOR
    # ==========================
    pkill -9 -f "cpu_monitor.py"
    echo "⏹️  CPU monitor stopped."
    sleep 5
}

# Merge client logs and generate figures
post_process_results() {
    local output_dir=$1
    local work_dir=$2
    local venv_path=$3
    
    echo "🔄 Merging client logs..."
    cat "$output_dir"/client*.jsonl > "$output_dir/client.jsonl" 2>/dev/null || echo "No client logs to merge"
    
    echo "🎨 Generating figures..."
    
    echo "📈 Generating send gap fig..."
    "$venv_path/bin/python" "../../figures/draw_send_gap.py" "$output_dir/" --time-window=2.0
    
    echo "📊 Generating CDF figures..."
    "$venv_path/bin/python" "../../figures/draw_cdf.py" "$output_dir/"
    
    echo "📈 Generating cumulative request figures..."
    "$venv_path/bin/python" "../../figures/draw_cumu.py" "$output_dir/"
    
    echo "📈 Generating statistics figures..."
    "$venv_path/bin/python" "../../figures/analyze_statistics.py" "$output_dir/"
    "$venv_path/bin/python" "../../figures/analyze_statistics_smooth.py" "$output_dir/"
    "$venv_path/bin/python" "../../figures/analyze_statistics_smooth.py" "$output_dir/" --smooth-window 5 --instances 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
}

# Cleanup processes after experiment
cleanup_processes() {
    trap - INT TERM
    echo "🧹 Cleaning up processes..."
    
    # Local cleanup
    kill_processes_by_pattern "$VENV_PATH/bin/vllm" "vLLM"
    kill_processes_by_pattern "$VENV_PATH/bin/python3 -s" "Dead Python"
    kill_processes_by_pattern "router_v2" "router"
    kill_processes_by_pattern "smart_runner.py" "smart runner"
    kill_processes_by_pattern "$WORK_DIR/target/release/client" "client"
    pkill -9 -f "cpu_monitor.py"
    
    # Remote cleanup (only if we have remote machines)
    if [[ "$USE_REMOTE" == "True" ]] && [ ${#REMOTE_IPS[@]} -gt 0 ]; then
        echo "🧹 Cleaning up remote processes..."
        kill_remote_processes_by_pattern ".vllmflashinfer/bin/vllm" "remote vLLM"
        kill_remote_processes_by_pattern ".vllmflashinfer/bin/python3 -s" "Dead Python"
        kill_remote_processes_by_pattern "router_v2" "remote router"
        kill_remote_processes_by_pattern "smart_runner.py" "remote smart runner" 
        kill_remote_processes_by_pattern "/target/release/client" "remote client"
    fi
    
    sleep 5
}

# Sync remote results to storage
sync_remote_results() {
    if [[ "$USE_REMOTE" != "True" ]] || [ ${#REMOTE_IPS[@]} -eq 0 ]; then
        return 0
    fi
    
    echo "📤 Syncing remote results to storage..."
    
    for i in "${!REMOTE_IPS[@]}"; do
        remote_ip="${REMOTE_IPS[$i]}"
        remote_output_dir="${REMOTE_OUTPUT_DIRS[$i]}"
        store_remote_output_base="${STORE_REMOTE_OUTPUT_BASES[$i]}"
        
        echo "  📤 Syncing $remote_ip:$remote_output_dir to $store_remote_output_base/"
        
        ssh "-p${SSH_PORT}" "$remote_ip" "
            if [ -d '$remote_output_dir' ]; then
                mkdir -p '$store_remote_output_base'
                rsync -av '$remote_output_dir/' '$store_remote_output_base/'
                echo '✅ Remote results synced successfully'
            else
                echo '⚠️  Remote output directory not found: $remote_output_dir'
            fi
        "
    done
}

# -----------------------------------------------------------------------------
# Main Script Execution
# -----------------------------------------------------------------------------

# Parse command line arguments

trap cleanup_processes INT
trap cleanup_processes USR1
trap cleanup_processes TERM

POSITIONAL_ARGS=()
VERBOSE=false
USER_SPECIFIED_OUTPUT_DIR=false
USER_SPECIFIED_REMOTE_OUTPUT_DIR=false
USER_SPECIFIED_REMOTE_IPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v)
            VERBOSE=true
            shift
        ;;
        --no-backend)
            NO_BACKEND=true
            shift
        ;;
        --work-dir)
            WORK_DIR="$2"
            shift
            shift
        ;;
        --venv-path)
            VENV_PATH="$2"
            shift
            shift
        ;;
        --remote-output-dir)
            # Parse a single comma-separated string into an array
            IFS=' ' read -r -a REMOTE_OUTPUT_DIRS <<< "$2"
            USER_SPECIFIED_REMOTE_OUTPUT_DIR=true
            shift 2
            ;;
        --remote-ips)
            # Parse a single comma-separated string into an array
            IFS=' ' read -r -a REMOTE_IPS <<< "$2"
            USER_SPECIFIED_REMOTE_IPS=true
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            USER_SPECIFIED_OUTPUT_DIR=true
            shift
            shift
        ;;
        -h|--help)
            print_usage
            exit 0
        ;;
        -*|--*)
            echo "Unknown option $1"
            print_usage
            exit 1
        ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
        ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"


echo "FINAL POSITIONAL ARGS COUNT: $#"
echo "ARGS: $@"

# Validate required arguments
if [ "$#" -ne 4 ]; then
    echo "Error: Incorrect number of arguments."
    print_usage
    exit 1
fi

CONFIG1="$1"
CONFIG2="$2"
CONFIG3="$3"
POLICY="$4"

# Validate that configuration files exist
validate_file_exists "$CONFIG1" "Backend configuration"
validate_file_exists "$CONFIG2" "Router configuration"
validate_file_exists "$CONFIG3" "Client configuration"

# Validate that directories exist
validate_directory_exists "$WORK_DIR" "Work"
validate_directory_exists "$DATASET_DIR" "Dataset"

# Set up remote paths based on REMOTE_IPS
setup_remote_paths

# Construct features string
FEATURES="${POLICY}"

# Save current directory
PREV_DIR="$(pwd)"

# Change to work directory
cd "$WORK_DIR" || {
    echo "Error: Failed to enter directory $WORK_DIR"
    exit 1
}

# Build project components
build_project_components "$FEATURES" "$WORK_DIR"

# Return to previous directory
cd "$PREV_DIR"

# Setup output directory
# Only append policy suffix if user didn't specify custom output directories
if [ "$USER_SPECIFIED_OUTPUT_DIR" = false ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_${POLICY}"
fi

# Setup remote output directories
setup_remote_output_directories "${TIMESTAMP}" "$POLICY"

setup_output_directory "$OUTPUT_DIR" "$CONFIG1" "$CONFIG2" "$CONFIG3" "$FEATURES" "$POLICY" "$WORK_DIR"

# Kill any previous processes
cleanup_processes

# Clean up previous tmux session
cleanup_tmux_session "$SESSION_NAME"

# Launch experiment in tmux session
launch_experiment_session "$SESSION_NAME" "$WORK_DIR" "$VENV_PATH" "$CONFIG1" "$CONFIG2" "$CONFIG3" "$OUTPUT_BASE" "$OUTPUT_DIR" "$MODEL_PATH" "$DATASET_DIR" "$NO_BACKEND" "$TIME_IN_SEC"

# Cleanup processes
cleanup_processes

# Post-process results
post_process_results "$OUTPUT_DIR" "$WORK_DIR" "$VENV_PATH"

# Sync remote results to storage
# sync_remote_results

# Final message
echo "🎉 Experiment completed successfully!"
echo "📁 Results are available in: $OUTPUT_DIR"
if [ ${#REMOTE_OUTPUT_DIRS[@]} -gt 0 ]; then
    echo "💾 Remote results are stored in:"
    for i in "${!REMOTE_OUTPUT_DIRS[@]}"; do
        echo "   - ${REMOTE_IPS[$i]}: ${REMOTE_OUTPUT_DIRS[$i]}"
    done
fi
echo "🔍 You can inspect the tmux session using: tmux attach-session -t $SESSION_NAME"