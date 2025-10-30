#!/bin/bash

# =============================================================================
# Metrics Test Runner - Grid Search QPS Script
# =============================================================================
# This script performs a grid search over different scaling factors by:
# 1. Generating client configuration files with different scale_factor values
# 2. Running experiments for each scaling factor and policy combination
# 3. Generating plots for each scaling factor
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration Section
# -----------------------------------------------------------------------------

# Define scaling factors to search over
SCALING_FACTORS=(3.0)

# Define batch sizes to test
BATCH_SIZES=(2048)

REMOTE_IPS="172.27.18.133"
SSH_PORT=10022

# Define policies to test
# "bounded-most-hit-q" is debugging now
# bounded-most-hit-q
# Each policies are about 
POLICIES=("round-robin-q" "least-wait-token-q" "bailian-impl-q" "join-shortest-q-weight" "join-shortest-q-tuple")

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../../config/dash-h20-1"
OUTPUT_BASE="/tmp/node1/lmmetric-logs"
REMOTE_OUTPUT_BASE="/tmp/node2/lmmetric-logs"
STORE_OUTPUT_BASE="/mnt/debugger/hjb/node1/lmmetric-logs"
STORE_REMOTE_OUTPUT_BASE="/mnt/debugger/hjb/node2/lmmetric-logs"


# Configuration files
ROUTER_CFG="$CONFIG_DIR/vllm_router.toml"
CLIENT_TEMPLATE="$CONFIG_DIR/bailian_clients.toml"

# -----------------------------------------------------------------------------
# Phase 1: Template Generation
# -----------------------------------------------------------------------------

echo "Phase 1: Generating TOML templates with different scaling factors..."

# Create directory for generated configs
GENERATED_CONFIGS_DIR="$SCRIPT_DIR/generated_configs"
mkdir -p "$GENERATED_CONFIGS_DIR"

# Generate TOML files for each scaling factor using the Python script
python3 "$SCRIPT_DIR/generate_toml_configs.py" \
    --template "$CLIENT_TEMPLATE" \
    --output-dir "$GENERATED_CONFIGS_DIR" \
    --scale-factors ${SCALING_FACTORS[@]}

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate TOML configurations"
    exit 1
fi

echo "Template generation completed successfully."


# -----------------------------------------------------------------------------
# Phase 2: Running Experiments
# -----------------------------------------------------------------------------

echo "Phase 2: Running experiments for each batch size, scaling factor, and policy..."

for bs in ${BATCH_SIZES[@]}; do
    TAG="batch${bs}_u0.9"
    BACKEND_CFG="$CONFIG_DIR/launch_vllm_16instances_b${bs}.toml"
    
    for sf in ${SCALING_FACTORS[@]}; do
        echo "Processing batch size: $bs, scaling factor: $sf"
        
        # Create directory for this batch size and scaling factor
        SF_DIR="$OUTPUT_BASE/${sf}_${TAG}"
        REMOTE_SF_DIR="$REMOTE_OUTPUT_BASE/${sf}_${TAG}"
        mkdir -p "$SF_DIR"
        
        # Get the generated config file for this scaling factor
        CLIENT_CFG="$GENERATED_CONFIGS_DIR/bailian_clients_sf${sf}.toml"
    
    if [ ! -f "$CLIENT_CFG" ]; then
        echo "Error: Client config file not found for scaling factor $sf"
        continue
    fi
    
    # Run experiments for each policy with retry logic
    for policy in ${POLICIES[@]}; do
        echo "Running experiment for policy: $policy"
        
        max_retries=3
        retry_count=0
        succeeded=false
        
        while [ $retry_count -lt $max_retries ]; do
            echo "Attempt $(($retry_count + 1))/$max_retries for scaling factor '$sf', policy '$policy'..."
            
            # Generate timestamp
            TIMESTAMP=$(date +%Y%m%d%H%M%S)
            
            # Create output directory for this experiment
            OUTPUT_DIR="$SF_DIR/${TIMESTAMP}_${policy}"
            REMOTE_OUTPUT_DIR="$REMOTE_SF_DIR/${TIMESTAMP}_${policy}"
            mkdir -p "$OUTPUT_DIR"
            
            # Run the experiment using bailian_dash2.sh
            if "$SCRIPT_DIR/bailian_dash.sh" \
                --output-dir "$OUTPUT_DIR" \
                --remote-output-dir "$REMOTE_OUTPUT_DIR" \
                "$BACKEND_CFG" \
                "$ROUTER_CFG" \
                "$CLIENT_CFG" \
                "$policy"; then
                echo "✅ Experiment completed successfully for scaling factor $sf, policy $policy"
                succeeded=true
                break
            else
                local exit_code=$?
                echo "❌ Experiment failed for scaling factor $sf, policy $policy with exit code $exit_code"
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

    echo "Moving TMP DIR to NFS"
    mv $SF_DIR $STORE_OUTPUT_BASE
    ssh -p "$SSH_PORT" "$REMOTE_IPS" "mv '${REMOTE_SF_DIR}' '${STORE_REMOTE_OUTPUT_BASE}'"
    done
done

echo "All experiments completed."

# -----------------------------------------------------------------------------
# Phase 3: Plotting Results
# -----------------------------------------------------------------------------

echo "Phase 3: Generating plots for each batch size and scaling factor..."

for bs in ${BATCH_SIZES[@]}; do
    TAG="batch${bs}_u0.9"
    
    for sf in ${SCALING_FACTORS[@]}; do
        echo "Generating plots for batch size: $bs, scaling factor: $sf"
        
        SF_DIR="$STORE_OUTPUT_BASE/${sf}_${TAG}"
    
    if [ ! -d "$SF_DIR" ]; then
        echo "Warning: Directory not found for scaling factor $sf"
        continue
    fi
    
    # TODO, draw compare CDF
    echo "Generating comparison CDF plot for scaling factor: $sf"
    
    # Declare associative array: policy -> latest_dir
    declare -A latest_dirs
    
    # Find all experiment directories for this scaling factor
    while IFS= read -r -d '' dir; do
        basename_dir=$(basename "$dir")
        # Match format: timestamp_policy (timestamp + underscore + policy name)
        if [[ $basename_dir =~ ^[0-9]+_([a-zA-Z0-9_-]+)$ ]]; then
            policy="${BASH_REMATCH[1]}"
            # Only process policies we care about
            case "$policy" in
                round-robin-q|least-wait-token-q|bailian-impl-q|join-shortest-q-weight|join-shortest-q-tuple)
                    # If this policy hasn't been recorded yet, or current timestamp is newer, update it
                    if [[ -z "${latest_dirs[$policy]}" ]] || [[ "$basename_dir" > "${latest_dirs[$policy]##*/}" ]]; then
                        latest_dirs[$policy]="$dir"
                    fi
                    ;;
            esac
        fi
    done < <(find "$SF_DIR" -mindepth 1 -maxdepth 1 -type d -print0)
    
    # Check if any valid directories were found
    if [ ${#latest_dirs[@]} -eq 0 ]; then
        echo "⚠️  No valid policy directories found in $SF_DIR"
    else
        # Sort policies for consistent order
        sorted_policies=()
        for policy in "${!latest_dirs[@]}"; do
            sorted_policies+=("$policy")
        done
        IFS=$'\n' sorted_policies=($(sort <<<"${sorted_policies[*]}"))
        unset IFS
        
        # Build directory list
        dir_list=()
        echo "📊 Found latest directories for policies:"
        for policy in "${sorted_policies[@]}"; do
            dir="${latest_dirs[$policy]}"
            echo "  - $policy: $dir"
            dir_list+=("$dir")
        done
        
        # Call compare_cdf.py
        echo "🎨 Generating comparison plot with compare_cdf.py..."
        python3 "$SCRIPT_DIR/../../figures/compare_cdf.py" "${dir_list[@]}"
        ret=$?
        if [ $ret -eq 0 ]; then
            echo "✅ Comparison plot generated successfully."
        else
            echo "❌ compare_cdf.py failed with exit code $ret"
        fi
    fi

    done
done

echo "All plotting completed."

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------

echo "Grid search QPS script completed successfully!"
echo "Results are organized by scaling factor in: $OUTPUT_BASE"
echo "Each scaling factor has its own directory (e.g., '0.5_${TAG}', '1.0_${TAG}', etc.)"
echo "Within each scaling factor directory, results are organized by timestamp and policy"

exit 0
