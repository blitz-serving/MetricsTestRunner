#!/bin/bash

# =============================================================================
# Metrics Test Runner - Grid Search QPS Script (Multi-Node Version)
# =============================================================================
# This script performs a grid search over different batch sizes and scaling factors by:
# 1. Generating client configuration files with different scale_factor values
# 2. Running experiments for each batch size, scaling factor, and policy combination
# 3. Generating plots for each batch size and scaling factor combination
# =============================================================================
# Updated: Supports multiple remote machines for distributed testing
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration Section
# -----------------------------------------------------------------------------

# Define scaling factors to search over
SCALING_FACTORS=(6.0)

# Define batch sizes to test
BATCH_SIZES=(4096)

# REMOTE_IPS is now an array to support multiple remote machines
REMOTE_IPS=("172.27.21.162" "172.27.21.163" "172.27.21.155")  # Add more IPs as needed
USE_REMOTE=True

SSH_PORT=10022

# Define policies to test
POLICIES=(
    "join-shortest-q-weight"
)

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../../config/dash-h20-1"

OUTPUT_BASE="/tmp/node1/lmmetric-logs"
STORE_OUTPUT_BASE="/mnt/debugger/hjb/node1/lmmetric-logs"

# REMOTE_OUTPUT_BASE and STORE_REMOTE_OUTPUT_BASE are now arrays
# Each remote machine should have its own output directories
REMOTE_OUTPUT_BASES=("/tmp/node2/lmmetric-logs" "/tmp/node3/lmmetric-logs"  "/tmp/node4/lmmetric-logs")  # Corresponds to REMOTE_IPS
STORE_REMOTE_OUTPUT_BASES=("/mnt/debugger/hjb/node2/lmmetric-logs" "/mnt/debugger/hjb/node3/lmmetric-logs" "/mnt/debugger/hjb/node4/lmmetric-logs")  # Corresponds to REMOTE_IPS

# Configuration files

CLIENT_TEMPLATE="$CONFIG_DIR/bailian_clients.toml"

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

# Function to validate remote configuration
validate_remote_config() {
    if [[ "$USE_REMOTE" == "True" ]]; then
        if [ ${#REMOTE_IPS[@]} -eq 0 ]; then
            echo "ERROR: USE_REMOTE is True but REMOTE_IPS array is empty" >&2
            exit 1
        fi
        
        if [ ${#REMOTE_IPS[@]} -ne ${#REMOTE_OUTPUT_BASES[@]} ] || [ ${#REMOTE_IPS[@]} -ne ${#STORE_REMOTE_OUTPUT_BASES[@]} ]; then
            echo "ERROR: Remote configuration mismatch. All remote arrays must have the same length." >&2
            echo "REMOTE_IPS count: ${#REMOTE_IPS[@]}"
            echo "REMOTE_OUTPUT_BASES count: ${#REMOTE_OUTPUT_BASES[@]}"
            echo "STORE_REMOTE_OUTPUT_BASES count: ${#STORE_REMOTE_OUTPUT_BASES[@]}"
            exit 1
        fi
        
        # Validate remote directory paths
        for i in "${!REMOTE_IPS[@]}"; do
            remote_ip="${REMOTE_IPS[$i]}"
            remote_output_base="${REMOTE_OUTPUT_BASES[$i]}"
            store_remote_output_base="${STORE_REMOTE_OUTPUT_BASES[$i]}"
            
            if [[ "$remote_output_base" != /tmp/* ]]; then
                echo "ERROR: Remote output base '$remote_output_base' for $remote_ip must be under /tmp/" >&2
                exit 1
            fi
            
            # Check if we can connect to the remote machine
            if ! ssh -p "$SSH_PORT" "$remote_ip" "echo 'Connection test successful'"; then
                echo "ERROR: Cannot connect to remote machine $remote_ip on port $SSH_PORT" >&2
                exit 1
            fi
        done
    fi
}

# Function to sync remote directories to storage
sync_remote_to_storage() {
    local sf="$1"
    local tag="$2"
    
    if [[ "$USE_REMOTE" != "True" ]]; then
        return 0
    fi
    
    echo "🔄 Syncing remote directories to storage for $sf_$tag..."
    
    for i in "${!REMOTE_IPS[@]}"; do
        remote_ip="${REMOTE_IPS[$i]}"
        remote_output_base="${REMOTE_OUTPUT_BASES[$i]}"
        store_remote_output_base="${STORE_REMOTE_OUTPUT_BASES[$i]}"
        
        remote_sf_dir="${remote_output_base}/${sf}_${tag}"
        echo "  📤 Syncing from $remote_ip:$remote_sf_dir to $store_remote_output_base/"
        
        ssh -p "$SSH_PORT" "$remote_ip" "
            if [[ ! -d '$remote_sf_dir' ]]; then
                echo '  ⚠️  Directory $remote_sf_dir does not exist on $remote_ip, skipping...'
                exit 0
            fi
            
            if [[ '$remote_sf_dir' != /tmp/* ]]; then
                echo '  ❌ ERROR: Remote directory $remote_sf_dir is not under /tmp/' >&2
                exit 1
            fi
            
            echo '  🔄 Starting rsync for $remote_sf_dir...'
            rsync -av '$remote_sf_dir' '$store_remote_output_base/' && rm -rf '$remote_sf_dir'
            echo '  ✅ Successfully synced and cleaned $remote_sf_dir'
        "
        
        if [ $? -ne 0 ]; then
            echo "  ❌ Failed to sync remote directory for $remote_ip" >&2
            return 1
        fi
    done
    
    echo "✅ All remote directories synced successfully"
    return 0
}

# Function to get remote output dirs array as string
get_remote_output_dirs_str() {
    local sf="$1"
    local tag="$2"
    local timestamp="$3"
    local policy="$4"
    
    local remote_dirs=()
    for i in "${!REMOTE_IPS[@]}"; do
        remote_output_base="${REMOTE_OUTPUT_BASES[$i]}"
        remote_dirs+=("${remote_output_base}/${sf}_${tag}/${timestamp}_${policy}")
    done
    
    echo "${remote_dirs[@]}"
}

# Function to output remote output dirs as separate words (for array capture)
get_remote_output_dirs() {
    local sf="$1"
    local tag="$2"
    local timestamp="$3"
    local policy="$4"
    
    local remote_dirs=()
    for i in "${!REMOTE_IPS[@]}"; do
        remote_output_base="${REMOTE_OUTPUT_BASES[$i]}"
        remote_dirs+=("${remote_output_base}/${sf}_${tag}/${timestamp}_${policy}")
    done
    
    # Output each element as a separate word
    printf '%s\n' "${remote_dirs[@]}"
}

# -----------------------------------------------------------------------------
# Phase 1: Template Generation
# -----------------------------------------------------------------------------

start=$SECONDS
echo "🚀 Phase 1: Generating TOML templates with different scaling factors..."

# Create directory for generated configs
GENERATED_CONFIGS_DIR="$SCRIPT_DIR/generated_configs"
# clean passed configs to prevent wrong running
rm -rf "$GENERATED_CONFIGS_DIR"
mkdir -p "$GENERATED_CONFIGS_DIR"

# Generate TOML files for each scaling factor using the Python script
python3 "$SCRIPT_DIR/generate_toml_configs.py" \
    --template "$CLIENT_TEMPLATE" \
    --output-dir "$GENERATED_CONFIGS_DIR" \
    --scale-factors ${SCALING_FACTORS[@]}

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to generate TOML configurations"
    exit 1
fi

echo "✅ Template generation completed successfully."

# Validate remote configuration before proceeding
validate_remote_config

# -----------------------------------------------------------------------------
# Phase 2: Running Experiments
# -----------------------------------------------------------------------------

echo "🔬 Phase 2: Running experiments for each batch size, scaling factor, and policy..."

for bs in ${BATCH_SIZES[@]}; do
    ROUTER_CFG="$CONFIG_DIR/vllm_router_new_fullargs_$bs.toml"
    for run_id in 1; do  # Run 3 times: r1, r2, r3
        TAG="batch${bs}_u0.9_flashinfer_1202_SCALE_r$run_id"
        
        if [[ "$USE_REMOTE" == "True" ]]; then
            BACKEND_CFG="$CONFIG_DIR/launch_vllm_16instances_b${bs}_4instances.toml"
        else
            echo "This should not happen"
            exit 1
            BACKEND_CFG="$CONFIG_DIR/launch_vllm_8instances_b${bs}_4instances.toml"
        fi

        for sf in ${SCALING_FACTORS[@]}; do
            echo "🎯 Processing batch size: $bs, scaling factor: $sf"
            
            # Create directory for this batch size and scaling factor
            SF_DIR="$OUTPUT_BASE/${sf}_${TAG}"
            mkdir -p "$SF_DIR"
            
            # Get the generated config file for this scaling factor
            CLIENT_CFG="$GENERATED_CONFIGS_DIR/bailian_clients_sf${sf}.toml"
        
            if [ ! -f "$CLIENT_CFG" ]; then
                echo "❌ Error: Client config file not found for scaling factor $sf"
                continue
            fi
        
            # Run experiments for each policy with retry logic
            for policy in ${POLICIES[@]}; do
                echo "🧪 Running experiment for policy: $policy"
                
                max_retries=5
                retry_count=0
                succeeded=false
                
                while [ $retry_count -lt $max_retries ]; do
                    echo "🔄 Attempt $(($retry_count + 1))/$max_retries for batch size $bs, scaling factor '$sf', policy '$policy'..."
                    
                    # Generate timestamp
                    TIMESTAMP=$(date +%Y%m%d%H%M%S)
                    
                    # Create output directory for this experiment
                    OUTPUT_DIR="$SF_DIR/${TIMESTAMP}_${policy}"
                    mkdir -p "$OUTPUT_DIR"
                    
                    # Get remote output directories as string
                    REMOTE_OUTPUT_DIRS_STR=$(get_remote_output_dirs_str "$sf" "$TAG" "$TIMESTAMP" "$policy")
                    REMOTE_IPS_STR="${REMOTE_IPS[@]}"  # 

                    echo "Calling with:"
                    echo "BACKEND_CFG=$BACKEND_CFG"
                    echo "ROUTER_CFG=$ROUTER_CFG"
                    echo "CLIENT_CFG=$CLIENT_CFG"
                    echo "policy=$policy"
                    echo "REMOTE_IPS: ${REMOTE_IPS_STR}"
                    echo "REMOTE_OUTPUT_DIRS_STR: $REMOTE_OUTPUT_DIRS_STR"
                    echo "\n\n"

                    # Run the experiment using bailian_dash_qwen30b.sh with multiple remote dirs
                    if "$SCRIPT_DIR/bailian_4instances.sh" \
                        --output-dir "$OUTPUT_DIR" \
                        --remote-output-dir "$REMOTE_OUTPUT_DIRS_STR" \
                        --remote-ips "$REMOTE_IPS_STR" \
                        "$BACKEND_CFG" \
                        "$ROUTER_CFG" \
                        "$CLIENT_CFG" \
                        "$policy"; then
                        echo "✅ Experiment completed successfully for batch size $bs, scaling factor $sf, policy $policy"
                        succeeded=true
                        break
                    else
                        exit_code=$?
                        echo "❌ Experiment failed for batch size $bs, scaling factor $sf, policy $policy with exit code $exit_code"
                        retry_count=$((retry_count + 1))
                        if [ $retry_count -lt $max_retries ]; then
                            echo "⏳ Retrying in 15 seconds..."
                            sleep 15
                        fi
                    fi
                done
                
                if [ "$succeeded" = false ]; then
                    echo "💥 Experiment failed after $max_retries attempts for batch size $bs, scaling factor $sf, policy $policy"
                fi
            done
            
            echo "🗄️  Moving TMP DIR to NFS for batch size $bs, scaling factor $sf"
            if [[ "$SF_DIR" != /tmp/* ]] || [[ ! -d "$SF_DIR" ]]; then
                echo "❌ ERROR: SF_DIR ('$SF_DIR') is not a valid directory under /tmp/" >&2
                exit 1
            fi
            
            # Sync local directory to storage
            echo "📤 Syncing local directory: $SF_DIR to $STORE_OUTPUT_BASE/"
            if rsync -av "$SF_DIR" "$STORE_OUTPUT_BASE/" && rm -rf "$SF_DIR"; then
                echo "✅ Local directory synced successfully"
            else
                echo "❌ Failed to sync local directory" >&2
                exit 1
            fi
            
            # Sync all remote directories to their respective storage locations
            if ! sync_remote_to_storage "$sf" "$TAG"; then
                echo "❌ Failed to sync remote directories" >&2
                exit 1
            fi
        done
    done
done

echo "✅ All experiments completed."

# -----------------------------------------------------------------------------
# Phase 3: Plotting Results
# -----------------------------------------------------------------------------

echo "📊 Phase 3: Generating plots for each batch size and scaling factor..."

for bs in ${BATCH_SIZES[@]}; do
    for run_id in 1; do  # Run 3 times: r1, r2, r3
        TAG="batch${bs}_u0.9_flashinfer_1202_SCALE_r$run_id"
        
        for sf in ${SCALING_FACTORS[@]}; do
            echo "📈 Generating plots for batch size: $bs, scaling factor: $sf"
            
            SF_DIR="$STORE_OUTPUT_BASE/${sf}_${TAG}"
        
            if [ ! -d "$SF_DIR" ]; then
                echo "⚠️  Warning: Directory not found for batch size $bs, scaling factor $sf at $SF_DIR"
                continue
            fi
            
            echo "🔍 Looking for experiment directories in: $SF_DIR"
            
            # Declare associative array: policy -> latest_dir
            unset latest_dirs  # Clear any previous data
            declare -A latest_dirs
        
            # Find all experiment directories for this batch size and scaling factor
            while IFS= read -r -d '' dir; do
                # Only process directories that contain client.jsonl
                if [ -f "$dir/client.jsonl" ]; then
                    basename_dir=$(basename "$dir")
                    # Match format: timestamp_policy (timestamp + underscore + policy name)
                    if [[ $basename_dir =~ ^[0-9]+_([a-zA-Z0-9_-]+)$ ]]; then
                        policy="${BASH_REMATCH[1]}"
                        # Only process policies we care about
                        if [[ -z "${latest_dirs[$policy]}" ]] || [[ "$basename_dir" > "${latest_dirs[$policy]##*/}" ]]; then
                            latest_dirs[$policy]="$dir"
                        fi
                    fi
                fi
            done < <(find "$SF_DIR" -mindepth 1 -maxdepth 1 -type d -print0)
        
            # Check if any valid directories were found
            if [ ${#latest_dirs[@]} -eq 0 ]; then
                echo "❌ No valid policy directories found in $SF_DIR"
                continue
            fi
            
            # Sort policies for consistent order
            sorted_policies=()
            for policy in "${!latest_dirs[@]}"; do
                sorted_policies+=("$policy")
            done
            IFS=$'\n' sorted_policies=($(sort <<<"${sorted_policies[*]}"))
            unset IFS
            
            # Build directory list
            dir_list=()
            echo "📋 Found latest directories for policies:"
            for policy in "${sorted_policies[@]}"; do
                dir="${latest_dirs[$policy]}"
                echo "  - $policy: $dir"
                dir_list+=("$dir")
            done
            
            # Generate comparison CDF plot
            TITLE_TAG="${sf}_${TAG}"
            echo "🎨 Generating comparison plot with title: $TITLE_TAG"
            python3 "$SCRIPT_DIR/../../figures/compare_cdf.py" "${dir_list[@]}" --label "$TITLE_TAG"
            ret=$?
            if [ $ret -eq 0 ]; then
                echo "✅ Comparison plot generated successfully."
            else
                echo "❌ compare_cdf.py failed with exit code $ret"
            fi
        done
    done
done

echo "✅ All plotting completed."

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------

echo "🎉 Grid search QPS script completed successfully!"
echo "📊 Executed phases: Template Generation, Experiment Running, Plotting"
echo "📁 Results are organized by batch size and scaling factor in: $STORE_OUTPUT_BASE"
echo "🔄 Remote results are stored in:"
for i in "${!REMOTE_IPS[@]}"; do
    remote_ip="${REMOTE_IPS[$i]}"
    store_remote_output_base="${STORE_REMOTE_OUTPUT_BASES[$i]}"
    echo "   - $remote_ip: $store_remote_output_base"
done

elapsed=$((SECONDS - start))
echo "⏱️  Total execution time: $elapsed seconds"

exit 0