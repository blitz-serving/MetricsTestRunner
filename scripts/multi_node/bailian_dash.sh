#!/bin/bash

# =============================================================================
# Metrics Test Runner - Bailian Evaluation Script (Modified Version)
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
# MODIFICATIONS:
# - Added --output-dir and --remote-output-dir options to override default paths
# - When these options are provided, they are used directly without ${TIMESTAMP}_${POLICY} suffix
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration Section - User Configurable Parameters
# -----------------------------------------------------------------------------

# Model path - directory containing the LLM model files
MODEL_PATH='/home/admin/resource/model/464482ce.Qwen2.5-7B-Instruct/1.0/'
REMOTE_MODEL_PATH='/home/admin/resource/model/464482ce.Qwen2.5-7B-Instruct/1.0/'

# Python virtual environment path with vLLM installed
VENV_PATH='/mnt/debugger/hjb/node1/yaullm/.venv'
REMOTE_VENV_PATH='/mnt/debugger/hjb/node2/yaullm/.venv'

# Project directory containing the blitz-infer-pack codebase
WORK_DIR='/mnt/debugger/hjb/node1/blitz-infer-pack'

# Flag to skip launching backend (useful for debugging)
NO_BACKEND=false

# Base directory for output logs
OUTPUT_BASE="/tmp/node1/lmmetric-logs"
REMOTE_OUTPUT_BASE="/tmp/node2/lmmetric-logs"
STORE_OUTPUT_BASE="/mnt/debugger/hjb/node1/lmmetric-logs"
STORE_REMOTE_OUTPUT_BASE="/mnt/debugger/hjb/node2/lmmetric-logs"

# Directory containing dataset files for client requests
DATASET_DIR="/mnt/debugger/hjb/node1/qwen-bailian-usagetraces-anon"

REMOTE_IPS="172.27.18.133"
SSH_PORT=10022

# Evaluation duration in seconds
# TODO, client need about 2min to fill the channel
TIME_IN_SEC=$((1200 + 120))

# Session name for tmux
SESSION_NAME="bailian"

# -----------------------------------------------------------------------------
# Derived Configuration - Computed from User Parameters
# -----------------------------------------------------------------------------

# Timestamped output directory for this run
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# These will be overridden if --output-dir or --remote-output-dir are provided
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_BASE}/${TIMESTAMP}"

# -----------------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------------

# Print usage information
print_usage() {
    echo "Usage: $0 [--no-backend] [--work-dir DIR] [--venv-path PATH] [--remote-output-dir DIR] [--remote-model-path PATH] [--remote-venv-path PATH] [-v] <backend-cfg> <router-cfg> <client-cfg> <policy>"
    echo "  policy must be one of: least-work-q, round-robin-q, join-shortest-q"
    echo ""
    echo "Options:"
    echo "  --no-backend          Skip launching vLLM backend processes"
    echo "  --work-dir DIR        Set working directory (default: $WORK_DIR)"
    echo "  --venv-path PATH      Set virtual environment path (default: $VENV_PATH)"
    echo "  --remote-output-dir DIR   Set remote output directory (default: $REMOTE_OUTPUT_DIR)"
    echo "  --remote-model-path PATH  Set remote model path (default: $REMOTE_MODEL_PATH)"
    echo "  --remote-venv-path PATH   Set remote virtual environment path (default: $REMOTE_VENV_PATH)"
    echo "  --output-dir DIR      Set local output directory (overrides default timestamped path)"
    echo "  --remote-output-dir DIR   Set remote output directory (overrides default timestamped path)"
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

# Kill processes by pattern
kill_processes_by_pattern() {
    local pattern=$1
    local description=$2
    echo "Killing $description processes..."
    pkill -f "$pattern" 2>/dev/null || true
}

# Kill processes by pattern on remote machines
kill_remote_processes_by_pattern() {
    local pattern=$1
    local description=$2
    local remote_ips=$3
    local remote_venv_path=$4
    
    echo "Killing $description processes on remote machines..."
    
    # Split remote_ips into an array
    IFS=',' read -ra IPS <<< "$remote_ips"
    
    # Kill processes on each remote machine
    for ip in "${IPS[@]}"; do
        echo "Killing processes on $ip..."
        ssh "-p ${SSH_PORT}" "$ip" "source $remote_venv_path/bin/activate && pkill -f -9 '$pattern' 2>/dev/null || true" 2>/dev/null || true
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
        cargo build -p router_v2  --features "$features"
        cargo build -p router_v2  --relase --features  "$features"
    else
        cargo build -p router_v2  --features "$features" --quiet
        cargo build -p router_v2  --release --features "$features" --quiet
    fi
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build router_v2."
        exit 1
    fi
    
    # Build request simulator client
    if [ "$VERBOSE" = true ]; then
        cargo build -p request-sim --bin client  -j64
        cargo build -p request-sim --release --bin client -j64
    else
        cargo build -p request-sim --bin client -j64 --quiet
        cargo build -p request-sim --release --bin client -j64 --quiet
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
    
    # Create output directory (policy suffix already handled in main logic)
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"
    
    # Copy configuration files to output directory
    cp "$config1" "$output_dir/backend.toml"
    cp "$config2" "$output_dir/router.toml"
    cp "$config3" "$output_dir/client.toml"
    
    # Save features string to commands.txt
    echo "$features" > "$output_dir/commands.txt"
    
    echo "$output_dir"
}

wait_for_vllm_startup() {
    local remote_dir="$OUTPUT_DIR"
    local max_wait_sec=240 # 4 minutes
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
    local remote_dir="$REMOTE_OUTPUT_DIR"
    local max_wait_sec=30 # the remote vllm should already started up
    local elapsed=0
    local check_interval=5

    if [ -z "$remote_dir" ]; then
        echo "ERROR: REMOTE_OUTPUT_DIR is not set." >&2
        return 1
    fi

    if [ -z "$REMOTE_IPS" ]; then
        echo "ERROR: REMOTE_IPS is not set." >&2
        return 1
    fi

    # Split remote_ips into an array
    IFS=',' read -ra IPS <<< "$REMOTE_IPS"

    elapsed=0

    while [ $elapsed -lt $max_wait_sec ]; do
        local all_ready=true

        # Check each remote IP
        for ip in "${IPS[@]}"; do
            # Check if remote log files exist
            local remote_log_check=$(ssh "-p ${SSH_PORT}" "$ip" "ls ${remote_dir}/vllm*.log 2>/dev/null" 2>/dev/null)
            
            if [ -z "$remote_log_check" ]; then
                echo "No vllm*.log files found in $remote_dir on $ip. Waiting..."
                all_ready=false
                break
            fi

            # Check each log file on the remote machine
            for remote_logfile in $remote_log_check; do
                local expected="INFO:     Application startup complete."
                local found=$(ssh "-p ${SSH_PORT}" "$ip" "grep -F '$expected' '$remote_logfile' 2>/dev/null" 2>/dev/null)
                
                if [ -z "$found" ]; then
                    echo "Waiting for $remote_logfile on $ip to complete startup..."
                    all_ready=false
                    break 2  # Break out of both loops
                fi
            done
        done

        if [ "$all_ready" = true ]; then
            echo "All remote vllm*.log files indicate startup complete."
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
    local remote_output_dir=${13}
    local remote_model_path=${14}
    local remote_venv_path=${15}
    
    # Source Python virtual environment activation command
    local tmux_cmd="source $venv_path/bin/activate"
    
    # Create new tmux session
    tmux new-session -d -s "$session_name"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Change to that directory
    cd "$SCRIPT_DIR" || exit 1

    # Launch vLLM backends if not skipped

    if [ "$no_backend" = false ]; then
        echo "Launching vLLM backends and waiting 120s..."
        tmux new-window -t "$session_name" -n window1
        tmux send-keys -t "$session_name:window1" "$tmux_cmd && python ../../smart_runner.py --toml $config1 --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --remote-output-dir=$remote_output_dir --remote-model-path=$remote_model_path --remote-venv-path=$remote_venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
        # Wait for vLLM startup with timeout
        if ! wait_for_vllm_startup; then
            echo "FATAL: vLLM failed to start in time. Aborting experiment." >&2
            cleanup_processes
            exit 1
        fi
        # Wait for remote vLLM startup instead of sleeping
        
        if ! wait_for_remote_vllm_startup; then
            echo "FATAL: Remote vLLM failed to start in time. Aborting experiment." >&2
            cleanup_processes
            exit 1
        fi
    fi
    
    # Launch router
    echo "Launching router and waiting 20s..."
    tmux new-window -t "$session_name" -n window2
    tmux send-keys -t "$session_name:window2" "$tmux_cmd && python ../../smart_runner.py --toml $config2 --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
    sleep 20

    # Launch client
    echo "Launching client..."
    tmux new-window -t "$session_name" -n window3
    tmux send-keys -t "$session_name:window3" "$tmux_cmd && python ../../smart_runner.py --toml $config3 --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
    
    # Wait for experiment to complete
    echo "Running experiment for ${time_in_sec}s..."
    sleep $(($time_in_sec + 30))  # Extra time for pending requests
}

# Merge client logs and generate figures
post_process_results() {
    local output_dir=$1
    local work_dir=$2
    local venv_path=$3
    
    echo "Merging client logs..."
    cat "$output_dir"/client*.jsonl > "$output_dir/client.jsonl"
    
    echo "Generating overall figures..."
    "$venv_path/bin/python" "../../figures/final_figure_zero.py" --output-dir="$output_dir/" 

    echo "Generating send gap fig \\n"
    "$venv_path/bin/python" "../../figures/draw_send_gap.py" "$output_dir/" --time-window=2.0

    echo "Generating many cdf figs \\n"
    "$venv_path/bin/python" "../../figures/draw_cdf.py" "$output_dir/"

    echo "Generating req number with time figs \\n"
    "$venv_path/bin/python" "../../figures/draw_cumu.py" "$output_dir/"
}

# Cleanup processes after experiment
cleanup_processes() {
    trap - INT TERM
    echo "Cleaning up processes..."
    kill_processes_by_pattern "$VENV_PATH/bin/vllm" "vLLM"
    kill_processes_by_pattern "router_v2" "router"
    kill_processes_by_pattern "smart_runner.py" "smart runner"
    
    # Clean up remote processes if remote IPs are provided
    if [ -n "$REMOTE_IPS" ] && [ -n "$REMOTE_VENV_PATH" ]; then
        kill_remote_processes_by_pattern "$REMOTE_VENV_PATH/bin/vllm" "remote vLLM" "$REMOTE_IPS" "$REMOTE_VENV_PATH"
    fi
    
    sleep 5
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
            REMOTE_OUTPUT_DIR="$2"
            USER_SPECIFIED_REMOTE_OUTPUT_DIR=true
            shift
            shift
        ;;
        --remote-model-path)
            REMOTE_MODEL_PATH="$2"
            shift
            shift
        ;;
        --remote-venv-path)
            REMOTE_VENV_PATH="$2"
            shift
            shift
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

case "$POLICY" in
    round-robin-q|join-shortest-q|bounded-most-hit-q|least-wait-token-q|bailian-impl-q|join-shortest-q-weight|join-shortest-q-tuple)
        # Valid policy, do nothing
        ;;
    *)
        echo "Error: policy must be legal, got: '$POLICY'" >&2
        exit 1
        ;;
esac

# Validate that configuration files exist
validate_file_exists "$CONFIG1" "Backend configuration"
validate_file_exists "$CONFIG2" "Router configuration"
validate_file_exists "$CONFIG3" "Client configuration"

# Validate that directories exist
validate_directory_exists "$WORK_DIR" "Work"
validate_directory_exists "$DATASET_DIR" "Dataset"

# Construct features string
FEATURES="${POLICY}"
#FEATURES=""


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

if [ "$USER_SPECIFIED_REMOTE_OUTPUT_DIR" = false ]; then
    REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR}_${POLICY}"
fi

setup_output_directory "$OUTPUT_DIR" "$CONFIG1" "$CONFIG2" "$CONFIG3" "$FEATURES" "$POLICY"

# Kill any previous processes
echo "Cleaning up previous processes..."
kill_processes_by_pattern "$VENV_PATH/bin/vllm" "previous vLLM"
kill_processes_by_pattern "router_v2" "previous router"
kill_processes_by_pattern "$WORK_PATH/target/release/client" "previous client"
kill_remote_processes_by_pattern "$REMOTE_VENV_PATH/bin/vllm" "remote vLLM" "$REMOTE_IPS" "$REMOTE_VENV_PATH"

sleep 10

# Clean up previous tmux session
cleanup_tmux_session "$SESSION_NAME"

# Launch experiment in tmux session
launch_experiment_session "$SESSION_NAME" "$WORK_DIR" "$VENV_PATH" "$CONFIG1" "$CONFIG2" "$CONFIG3" "$OUTPUT_BASE" "$OUTPUT_DIR" "$MODEL_PATH" "$DATASET_DIR" "$NO_BACKEND" "$TIME_IN_SEC" "$REMOTE_OUTPUT_DIR" "$REMOTE_MODEL_PATH" "$REMOTE_VENV_PATH"

# Post-process results
post_process_results "$OUTPUT_DIR" "$WORK_DIR" "$VENV_PATH"

#echo "moving logs to nfs"
#mv $OUTPUT_DIR $STORE_OUTPUT_BASE
#ssh -p "$SSH_PORT" "$ip" "mv '${REMOTE_OUTPUT_DIR}' '${STORE_REMOTE_OUTPUT_BASE}'"

# Cleanup processes
cleanup_processes

# Final message
echo "Experiment completed successfully!"
echo "Results are available in: $OUTPUT_DIR"
echo "You can inspect the tmux session using: tmux attach-session -t $SESSION_NAME"
