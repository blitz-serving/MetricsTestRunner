#!/bin/bash

# =============================================================================
# Metrics Test Runner - Selective Azure Evaluation Script
# =============================================================================
# Supports selectively launching components: backend, router, client.
# =============================================================================

# ----------------------------------------------------------------------------- 
# User Configurable Parameters
# -----------------------------------------------------------------------------

MODEL_PATH='/nvme/models/Qwen2.5-7B-Instruct'
VENV_PATH='/nvme/zkx/modified-vllm/myenv'          # ⚠️ 不带 /bin
WORK_DIR='/nvme/zkx/blitz-infer-pack'
OUTPUT_BASE="/nvme/lmetric/logs/lmmetric-logs"
DATASET_DIR="/nvme/lmetric/datasets"
TIME_IN_SEC=120
SESSION_NAME="azure"

# Components to run (default: all)
COMPONENTS="backend,router,client"

# ----------------------------------------------------------------------------- 
# Derived Configuration
# -----------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"

# ----------------------------------------------------------------------------- 
# Helper Functions
# -----------------------------------------------------------------------------

print_usage() {
    echo "Usage: $0 [--components LIST] [--work-dir DIR] [--venv-path PATH] <backend-cfg> <router-cfg> <client-cfg> <policy>"
    echo ""
    echo "Components (comma-separated): backend, router, client"
    echo "  e.g. --components backend,router or --components client"
    echo ""
    echo "Options:"
    echo "  --components LIST  Components to launch (default: backend,router,client)"
    echo "  --work-dir DIR     Set working directory (default: $WORK_DIR)"
    echo "  --venv-path PATH   Set virtual environment path (default: $VENV_PATH)"
    echo "  -h, --help         Show this help message"
}

validate_file_exists() {
    local file_path=$1
    local description=$2
    if [ ! -f "$file_path" ]; then
        echo "Error: $description file '$file_path' not found."
        exit 1
    fi
}

validate_directory_exists() {
    local dir_path=$1
    local description=$2
    if [ ! -d "$dir_path" ]; then
        echo "Error: $description directory '$dir_path' not found."
        exit 1
    fi
}

kill_processes_by_pattern() {
    local pattern=$1
    local description=$2
    echo "Killing $description processes..."
    pkill -f "$pattern" 2>/dev/null || true
}

cleanup_tmux_session() {
    local session_name=$1
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Cleaning up existing '$session_name' tmux session..."
        for pane in $(tmux list-panes -t "$session_name" -F '#{pane_id}'); do
            tmux send-keys -t "$pane" C-c
        done
        sleep 2
        tmux kill-session -t "$session_name"
    fi
}

build_project_components() {
    local features=$1
    echo "Building router_v2 (features: $features)..."
    cargo build -p router_v2 --features "$features"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build router_v2."
        exit 1
    fi
    echo "Building request-sim client..."
    cargo build -p request-sim --bin client --release -j64
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build request-sim client."
        exit 1
    fi
    echo "Build successful."
}

setup_output_directory() {
    local output_dir=$1
    local config1=$2
    local config2=$3
    local config3=$4
    local features=$5
    local policy=$6

    mkdir -p "$output_dir"
    cp "$config1" "$output_dir/backend.toml"
    cp "$config2" "$output_dir/router.toml"
    cp "$config3" "$output_dir/client.toml"
    echo "$features" > "$output_dir/commands.txt"
}

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
    local time_in_sec=${11}
    local components=${12}

    local tmux_cmd="source $venv_path/bin/activate"
    tmux new-session -d -s "$session_name"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR" || exit 1

    local comps_lower=$(echo "$components" | tr '[:upper:]' '[:lower:]')

    if [[ "$comps_lower" == *"backend"* ]]; then
        echo "Launching backend..."
        tmux new-window -t "$session_name" -n backend
        tmux send-keys -t "$session_name:backend" "$tmux_cmd && python ../../smart_runner.py --toml $config1 --log-dir=$output_base --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
        # sleep 120
    fi

    if [[ "$comps_lower" == *"router"* ]]; then
        echo "Launching router..."
        tmux new-window -t "$session_name" -n router
        echo "python ../../smart_runner.py --toml $config2 --log-dir=$output_base --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir"
        
        tmux send-keys -t "$session_name:router" "$tmux_cmd && python ../../smart_runner.py --toml $config2 --log-dir=$output_base --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
        echo "python ../../smart_runner.py --toml $config2 --log-dir=$output_base --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir"
        # sleep 20
    fi

    if [[ "$comps_lower" == *"client"* ]]; then
        echo "Launching client..."
        tmux new-window -t "$session_name" -n client
        tmux send-keys -t "$session_name:client" "$tmux_cmd && python ../../smart_runner.py --toml $config3 --log-dir=$output_base --output-dir=$output_dir --model-path=$model_path --venv-path=$venv_path --work-dir=$work_dir --dataset-dir=$dataset_dir" C-m
    fi

    # echo "Running experiment for ${time_in_sec}s..."
    # sleep $(($time_in_sec + 30))
}

post_process_results() {
    local output_dir=$1
    local venv_path=$2
    echo "Merging client logs..."
    if ls "$output_dir"/client*.jsonl 1>/dev/null 2>&1; then
        cat "$output_dir"/client*.jsonl > "$output_dir/client.jsonl"
    fi
    echo "Generating overall figures..."
    "$venv_path/bin/python" "../../figures/final_figure_zero.py" --output-dir="$output_dir/"
}

cleanup_processes() {
    local venv_path=$1
    echo "Cleaning up processes..."
    # kill_processes_by_pattern "$venv_path/bin/vllm" "vLLM"
    # kill_processes_by_pattern "router_v2" "router"
    # kill_processes_by_pattern "smart_runner.py" "smart runner"
    sleep 5
}

# ----------------------------------------------------------------------------- 
# Main Execution
# -----------------------------------------------------------------------------

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --components)
            COMPONENTS="$2"
            shift
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

if [ "$#" -ne 4 ]; then
    echo "Error: Incorrect number of arguments."
    print_usage
    exit 1
fi

CONFIG1="$1"
CONFIG2="$2"
CONFIG3="$3"
POLICY="$4"

# VALIDATE_POLICY=("least-work-q" "round-robin-q" "join-shortest-q")
# if [[ ! " ${VALIDATE_POLICY[*]} " =~ " ${POLICY} " ]]; then
#     echo "Error: policy must be one of: ${VALIDATE_POLICY[*]}, got: '$POLICY'"
#     exit 1
# fi

validate_file_exists "$CONFIG1" "Backend configuration"
validate_file_exists "$CONFIG2" "Router configuration"
validate_file_exists "$CONFIG3" "Client configuration"
validate_directory_exists "$WORK_DIR" "Work"
validate_directory_exists "$DATASET_DIR" "Dataset"

FEATURES="${POLICY}"
PREV_DIR="$(pwd)"
cd "$WORK_DIR" || exit 1
build_project_components "$FEATURES"
cd "$PREV_DIR"

OUTPUT_DIR="${OUTPUT_DIR}_${POLICY}"
setup_output_directory "$OUTPUT_DIR" "$CONFIG1" "$CONFIG2" "$CONFIG3" "$FEATURES" "$POLICY"

echo "Cleaning up previous processes..."
# kill_processes_by_pattern "$VENV_PATH/bin/vllm" "previous vLLM"
# kill_processes_by_pattern "router_v2" "previous router"
# kill_processes_by_pattern "$WORK_DIR/target/release/client" "previous client"
sleep 5
# cleanup_tmux_session "$SESSION_NAME"

launch_experiment_session "$SESSION_NAME" "$WORK_DIR" "$VENV_PATH" "$CONFIG1" "$CONFIG2" "$CONFIG3" "$OUTPUT_BASE" "$OUTPUT_DIR" "$MODEL_PATH" "$DATASET_DIR" "$TIME_IN_SEC" "$COMPONENTS"

post_process_results "$OUTPUT_DIR" "$VENV_PATH"
# cleanup_processes "$VENV_PATH"

echo "Experiment completed successfully!"
echo "Results are available in: $OUTPUT_DIR"
echo "You can inspect the tmux session using: tmux attach -t $SESSION_NAME"
