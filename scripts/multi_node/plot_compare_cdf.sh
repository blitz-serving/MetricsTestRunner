#!/bin/bash

# =============================================================================
# Metrics Test Runner - Grid Search QPS Script (Corrected Version)
# =============================================================================
# This script performs a grid search over different batch sizes and scaling factors by:
# 1. Generating client configuration files with different scale_factor values
# 2. Running experiments for each batch size, scaling factor, and policy combination
# 3. Generating plots for each batch size and scaling factor combination
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration Section
# -----------------------------------------------------------------------------

# Define scaling factors to search over
SCALING_FACTORS=(5.0)

# Define batch sizes to test
BATCH_SIZES=(1024 4096)

REMOTE_IPS="172.27.18.231"
SSH_PORT=10022

# Define policies to test
# "bounded-most-hit-q" is debugging now
# bounded-most-hit-q
# Each policies are about 
#POLICIES=("round-robin-q" "random-q" "bailian-impl-q" "join-shortest-q-weight" "join-shortest-q-tuple")
# Noted that bound-mosthit-q can only run 1024 now.. "least-wait-token-q" "bounded-most-hit-q"
POLICIES=("round-robin-q" "random-q" "bailian-impl-00" "bailian-impl-01" "bailian-impl-02" "bailian-impl-03" "bailian-impl-04" "bailian-impl-05" "bailian-impl-06" "bailian-impl-07" "bailian-impl-08" "bailian-impl-09" "bailian-impl-10" "least-wait-token-q" "join-shortest-q-weight")
#POLICIES=("bailian-impl-q" "bailian-impl-kv" "bailian-impl-rqs" "bailian-impl-tks")


# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../../config/dash-h20-1"
OUTPUT_BASE="/tmp/node1/lmmetric-logs"
REMOTE_OUTPUT_BASE="/tmp/node2/lmmetric-logs"
STORE_OUTPUT_BASE="/mnt/debugger/hjb/node1/lmmetric-logs"
STORE_REMOTE_OUTPUT_BASE="/mnt/debugger/hjb/node2/lmmetric-logs"


# Configuration files
ROUTER_CFG="$CONFIG_DIR/vllm_router.toml"
#CLIENT_TEMPLATE="$CONFIG_DIR/bailian_clients.toml"
CLIENT_TEMPLATE="$CONFIG_DIR/bailian_clients.toml" # TraceB

# -----------------------------------------------------------------------------
# Phase 3: Plotting Results
# -----------------------------------------------------------------------------

echo "Phase 3: Generating plots for each batch size and scaling factor..."

for bs in ${BATCH_SIZES[@]}; do
    TAG="batch${bs}_u0.9_ugly4"
    
    for sf in ${SCALING_FACTORS[@]}; do
        echo "Generating plots for batch size: $bs, scaling factor: $sf"
        
        SF_DIR="$STORE_OUTPUT_BASE/${sf}_${TAG}"
    
        if [ ! -d "$SF_DIR" ]; then
            echo "Warning: Directory not found for batch size $bs, scaling factor $sf"
            continue
        fi
        echo "try to finding at dir ${SF_DIR}"
        # Generate comparison CDF plot for batch size $bs, scaling factor: $sf
        TITLE_TAG=${sf}_${TAG}
        # Declare associative array: policy -> latest_dir
        unset latest_dirs  # Clear any previous data
        declare -A latest_dirs
    
        # Find all experiment directories for this batch size and scaling factor
        # Only search within the specific SF_DIR, do not look elsewhere
        while IFS= read -r -d '' dir; do
            # Only process directories that contain client.jsonl
            if [ -f "$dir/client.jsonl" ]; then
                basename_dir=$(basename "$dir")
                # Match format: timestamp_policy (timestamp + underscore + policy name)
                if [[ $basename_dir =~ ^[0-9]+_([a-zA-Z0-9_-]+)$ ]]; then
                    policy="${BASH_REMATCH[1]}"
                    # Only process policies we care about
                    case "$policy" in
                        round-robin-q|least-wait-token-q|bailian-impl-q|join-shortest-q-weight|join-shortest-q-tuple|random-q|bailian-impl-kv|bailian-impl-rqs|bailian-impl-tks|bailian-impl-00|bailian-impl-01|bailian-impl-02|bailian-impl-03|bailian-impl-04|bailian-impl-05|bailian-impl-06|bailian-impl-07|bailian-impl-08|bailian-impl-09|bailian-impl-10)
                            # If this policy hasn't been recorded yet, or current timestamp is newer, update it
                            if [[ -z "${latest_dirs[$policy]}" ]] || [[ "$basename_dir" > "${latest_dirs[$policy]##*/}" ]]; then
                                latest_dirs[$policy]="$dir"
                            fi
                            ;;
                    esac
                fi
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
            echo "🎨 Generating comparison plot with compare_cdf.py... ${TITLE_TAG}"
            python3 "$SCRIPT_DIR/../../figures/compare_cdf.py" "${dir_list[@]}" --label "${TITLE_TAG}"
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
echo "Executed phases: $PHASES"
echo "Results are organized by batch size and scaling factor in: $OUTPUT_BASE"
echo "Each combination has its own directory (e.g., '2.5_batch2048_u0.9', '3.0_batch3072_u0.9', etc.)"
echo "Within each directory, results are organized by timestamp and policy"

exit 0
