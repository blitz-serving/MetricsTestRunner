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
SCALING_FACTORS=(2.0)
MACHINE="12"
NODE1="1"
NODE2="2"

# Define batch sizes to test
BATCH_SIZES=(4096)

USE_REMOTE=True # True

SSH_PORT=10022
tag="flashinfer_1205_try_scale_mooncake_tool_r1"

# /mnt/debugger/hjb/node1/lmmetric-logs/2.2_batch4096_u0.9_flashinfer_1205_coder_scaling_qwen30b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/1.5_batch4096_u0.9_flashinfer_1202_mooncake_conv_qwen7b_r1
# /mnt/debugger/hjb/node3/lmmetric-logs/1.0_batch4096_u0.9_flashinfer_1204_mooncake_tool_qwen30b_fix_timeout_length_r1
#/mnt/debugger/hjb/node3/lmmetric-logs/1.5_batch4096_u0.9_flashinfer_1204_mooncake_conv_final_fix_qwen30b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_1119_lwl_gated_param_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/5.6_batch4096_u0.9_flashinfer_1203_toC_scaling_30b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/5.9_batch4096_u0.9_flashinfer_1205_scaling_toC_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/2.0_batch4096_u0.9_flashinfer_1205_try_scale_mooncake_tool_r1

# Define policies to test
# "bounded-most-hit-q" is debugging now
# bounded-most-hit-q
# Each policies are about 
#POLICIES=("round-robin-q" "random-q" "bailian-impl-q" "join-shortest-q-weight" "join-shortest-q-tuple")
# Noted that bound-mosthit-q can only run 1024 now.. "least-wait-token-q" "bounded-most-hit-q"
#POLICIES=("round-robin-q" "random-q" "bailian-impl-00" "bailian-impl-01" "bailian-impl-02" "bailian-impl-03" "bailian-impl-04" "bailian-impl-05" "bailian-impl-06" "bailian-impl-07" "bailian-impl-08" "bailian-impl-09" "bailian-impl-10" "least-wait-token-q" "join-shortest-q-weight")
#POLICIES=("round-robin-q" "dynamo-deterministic" "least-wait-token-random" "least-wait-token-q" "least-wait-token-bs" "bailian-impl-06" "join-shortest-q-weight" "join-shortest-q-tuple")
#POLICIES=("kvhit-tpot")
#POLICIES=("dynamo-deterministic" "least-wait-token-random" "least-wait-token-bs" "bailian-impl-lwl-00" "bailian-impl-lwl-01" "bailian-impl-lwl-02" "bailian-impl-lwl-03" "bailian-impl-lwl-04" "bailian-impl-lwl-05" "bailian-impl-lwl-06" "bailian-impl-lwl-07" "bailian-impl-lwl-08" "bailian-impl-lwl-09" "bailian-impl-lwl-10")
#POLICIES=("bailian-impl-06" "dynamo-deterministic" "least-wait-token-bs")
#POLICIES=("bailian-tuple" "llumnix-tuple" "bailian-impl-06" "llumnix-linear-04")
# 20x30min = 10h;
# 4 * 3  / 2 = 6h;
POLICIES=(
    "join-shortest-q-weight"
    "bailian-impl-06"
    #"bailian-impl-05-deterministic"
    "dynamo-deterministic"
    # "least-wait-token-gated-bs"
    "least-wait-token-mul-bs"
    "join-shortest-q-ttft"
    # "llmd-impl-q"
)
# "least-wait-token-mul-bs-sample"

#"llmd-impl-q"
# "poly-serve-impl-q"
# "slo-serve-impl-q"
#POLICIES=("least-wait-token-q" "least-wait-token-random" "llmd-impl-q")
#POLICIES=("dynamo-deterministic" "least-wait-token-gated-bs" "least-wait-token-bs")
# TODO, new ttft+bs policy
# "llmd-impl-q" "poly-serve-impl-q" "slo-serve-impl-q" "least-ttft-bs-tuple"
#POLICIES=("least-wait-token-random")

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../../config/dash-h20-1"
SIM_CONFIG_DIR="$SCRIPT_DIR/../../config/ipads-h20-1"
OUTPUT_BASE="/tmp/node${NODE1}/lmmetric-logs"
REMOTE_OUTPUT_BASE="/tmp/node${NODE2}/lmmetric-logs"
STORE_OUTPUT_BASE="/mnt/debugger/hjb/node${NODE1}/lmmetric-logs"
STORE_REMOTE_OUTPUT_BASE="/mnt/debugger/hjb/node${NODE2}/lmmetric-logs"

# -----------------------------------------------------------------------------
# Phase 3: Plotting Results
# -----------------------------------------------------------------------------

echo "Phase 3: Generating plots for each batch size and scaling factor..."

for bs in ${BATCH_SIZES[@]}; do
    for run_id in 1; do  # Run 3 times: r1, r2, r3
        TAG="batch${bs}_u0.9_${tag}"
        
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
        
            POLICIES_STR=" ${POLICIES[*]} "
            while IFS= read -r -d '' dir; do
                if [ -f "$dir/client.jsonl" ]; then
                    basename_dir=$(basename "$dir")
                    if [[ $basename_dir =~ ^[0-9]+_([a-zA-Z0-9_-]+)$ ]]; then
                        policy="${BASH_REMATCH[1]}"
                        # Only consider policies in the POLICIES array
                        if [[ " ${POLICIES_STR} " == *" $policy "* ]]; then
                            if [[ -z "${latest_dirs[$policy]}" ]] || [[ "$basename_dir" > "${latest_dirs[$policy]##*/}" ]]; then
                                latest_dirs[$policy]="$dir"
                            fi
                        fi
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

elapsed=$((SECONDS - start))
echo "执行耗时: $elapsed 秒"

exit 0
