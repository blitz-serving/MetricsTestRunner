#!/bin/bash

# =============================================================================
# Metrics Test Runner - Azure Evaluation Script
# =============================================================================
# This script orchestrates distributed LLM inference experiments using:
# 1. vLLM backends (model serving)
# 2. Router (request routing)
# 3. Client (request generation)
#
# The script creates a tmux session with three windows for each component,
# runs the experiment for a specified duration, collects logs, and generates
# performance visualization figures.
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration Section - User Configurable Parameters
# -----------------------------------------------------------------------------

# Model path - directory containing the LLM model files
MODEL_PATH='/home/hanjinbo.hjb/Qwen2.5-7B-Instruct'
REMOTE_MODEL_PATH='/home/hanjinbo.hjb/Qwen2.5-7B-Instruct'

# Python virtual environment path with vLLM installed
VENV_PATH='/home/hanjinbo.hjb/yaullm/.venv'
REMOTE_VENV_PATH='/home/hanjinbo.hjb/yaullm/.venv'

# Project directory containing the blitz-infer-pack codebase
WORK_DIR='/home/hanjinbo.hjb/blitz-infer-pack'

# Flag to skip launching backend (useful for debugging)
NO_BACKEND=false

# Base directory for output logs
OUTPUT_BASE="/home/hanjinbo.hjb/lmmetric-logs"
REMOTE_OUTPUT_BASE="/home/hanjinbo.hjb/lmmetric-logs"

# Directory containing dataset files for client requests
DATASET_DIR="/home/hanjinbo.hjb/AzurePublicDataset/data"

REMOTE_IPS="33.254.60.218"

# Evaluation duration in seconds
TIME_IN_SEC=1200

# Session name for tmux
SESSION_NAME="azure"

# -----------------------------------------------------------------------------
# Derived Configuration - Computed from User Parameters
# -----------------------------------------------------------------------------

# Timestamped output directory for this run
TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_BASE}/${TIMESTAMP}"
# -----------------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------------

# Print usage information
print_usage() {
    echo "Usage: $0 [--no-backend] [--work-dir DIR] [--venv-path PATH] [--remote-output-dir DIR] [--remote-model-path PATH] [--remote-venv-path PATH] <backend-cfg> <router-cfg> <client-cfg> <policy>"
    echo "  policy must be one of: least-work-q, round-robin-q, join-shortest-q"
    echo ""
    echo "Options:"
    echo "  --no-backend          Skip launching vLLM backend processes"
    echo "  --work-dir DIR        Set working directory (default: $WORK_DIR)"
    echo "  --venv-path PATH      Set virtual environment path (default: $VENV_PATH)"
    echo "  --remote-output-dir DIR   Set remote output directory (default: $REMOTE_OUTPUT_DIR)"
    echo "  --remote-model-path PATH  Set remote model path (default: $REMOTE_MODEL_PATH)"
    echo "  --remote-venv-path PATH   Set remote virtual environment path (default: $REMOTE_VENV_PATH)"
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
        ssh "$ip" "source $remote_venv_path/bin/activate && pkill -f '$pattern' 2>/dev/null || true" 2>/dev/null || true
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
    # TODO, note this is debug mode now --release 
    cargo build -p router_v2 --features "$features"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build router_v2."
        exit 1
    fi
    
    # Build request simulator client
    cargo build -p request-sim --bin client --release -j64
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
    
    # Create timestamped output directory with policy suffix
    local policy_output_dir="${output_dir}"
    echo "Creating output directory: $policy_output_dir"
    mkdir -p "$policy_output_dir"
    
    # Copy configuration files to output directory
    cp "$config1" "$policy_output_dir/backend.toml"
    cp "$config2" "$policy_output_dir/router.toml"
    cp "$config3" "$policy_output_dir/client.toml"
    
    # Save features string to commands.txt
    echo "$features" > "$policy_output_dir/commands.txt"
    
    echo "$policy_output_dir"
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
        tmux send-keys -t "$session_name:window1" "$tmux_cmd && python ../../smart_runner_v2.py --toml $config1 --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --remote-output-dir=$remote_output_dir --remote-model-path=$remote_model_path --remote-venv-path=$remote_venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
        sleep 120
    fi
    
    # Launch router
    echo "Launching router and waiting 20s..."
    tmux new-window -t "$session_name" -n window2
    tmux send-keys -t "$session_name:window2" "$tmux_cmd && python ../../smart_runner_v2.py --toml $config2 --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
    sleep 20
    
    sleep 5000
    # # Launch client
    # echo "Launching client..."
    # tmux new-window -t "$session_name" -n window3
    # tmux send-keys -t "$session_name:window3" "$tmux_cmd && python ../../smart_runner_v2.py --toml $config3 --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
    
    # # Wait for experiment to complete
    # echo "Running experiment for ${time_in_sec}s..."
    # sleep $(($time_in_sec + 30))  # Extra time for pending requests
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
}

# Cleanup processes after experiment
cleanup_processes() {
    local venv_path=$1
    local remote_ips=$2
    local remote_venv_path=$3
    echo "Cleaning up processes..."
    kill_processes_by_pattern "$venv_path/bin/vllm" "vLLM"
    kill_processes_by_pattern "router_v2" "router"
    kill_processes_by_pattern "smart_runner.py" "smart runner"
    
    # Clean up remote processes if remote IPs are provided
    if [ -n "$remote_ips" ] && [ -n "$remote_venv_path" ]; then
        kill_remote_processes_by_pattern "$remote_venv_path/bin/vllm" "remote vLLM" "$remote_ips" "$remote_venv_path"
    fi
    
    sleep 5
}

# -----------------------------------------------------------------------------
# Main Script Execution
# -----------------------------------------------------------------------------

# Parse command line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Validate policy
if [[ "$POLICY" != "least-work-q" && "$POLICY" != "round-robin-q" && "$POLICY" != "join-shortest-q" ]]; then
    echo "Error: policy must be 'least-work-q' or 'round-robin-q' or 'join-shortest-q', got: '$POLICY'"
    exit 1
fi

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
OUTPUT_DIR="${OUTPUT_DIR}_${POLICY}"
setup_output_directory "$OUTPUT_DIR" "$CONFIG1" "$CONFIG2" "$CONFIG3" "$FEATURES" "$POLICY"

# Kill any previous processes
echo "Cleaning up previous processes..."
kill_processes_by_pattern "$VENV_PATH/bin/vllm" "previous vLLM"
kill_processes_by_pattern "router_v2" "previous router"
kill_processes_by_pattern "$WORK_PATH/target/release/client" "previous client"

sleep 10

# Clean up previous tmux session
cleanup_tmux_session "$SESSION_NAME"

# Launch experiment in tmux session
launch_experiment_session "$SESSION_NAME" "$WORK_DIR" "$VENV_PATH" "$CONFIG1" "$CONFIG2" "$CONFIG3" "$OUTPUT_BASE" "$OUTPUT_DIR" "$MODEL_PATH" "$DATASET_DIR" "$NO_BACKEND" "$TIME_IN_SEC" "$REMOTE_OUTPUT_DIR" "$REMOTE_MODEL_PATH" "$REMOTE_VENV_PATH"

# Post-process results
# post_process_results "$OUTPUT_DIR" "$WORK_DIR" "$VENV_PATH"

# Cleanup processes
cleanup_processes "$VENV_PATH" "$REMOTE_IPS" "$REMOTE_VENV_PATH"

# Final message
echo "Experiment completed successfully!"
echo "Results are available in: $OUTPUT_DIR"
echo "You can inspect the tmux session using: tmux attach-session -t $SESSION_NAME"
