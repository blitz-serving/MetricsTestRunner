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
BATCH_SIZES=(1024)

REMOTE_IPS="172.27.21.64"
USE_REMOTE=True # True

SSH_PORT=10022

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
POLICIES=("least-wait-token-bs")

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
#CLIENT_TEMPLATE="$CONFIG_DIR/bailian_clientb.toml" # TraceB
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

echo "Phase 2: Running experiments for each batch size, scaling factor, policy, and run ID..."

for bs in ${BATCH_SIZES[@]}; do
    for run_id in {1..3}; do  # Run 3 times: r1, r2, r3
        TAG="batch${bs}_u0.9_1118_lwl_bs_tuple_r${run_id}"
        
        if [[ "$USE_REMOTE" == "True" ]]; then
            BACKEND_CFG="$CONFIG_DIR/launch_vllm_16instances_b${bs}.toml"
        else
            BACKEND_CFG="$CONFIG_DIR/launch_vllm_8instances_b${bs}.toml"
        fi

        for sf in ${SCALING_FACTORS[@]}; do
            echo "Processing batch size: $bs, scaling factor: $sf, run: r${run_id}"
            
            # Create directory for this batch size, scaling factor, and run
            SF_DIR="$OUTPUT_BASE/${sf}_${TAG}"
            REMOTE_SF_DIR="$REMOTE_OUTPUT_BASE/${sf}_${TAG}"
            mkdir -p "$SF_DIR"
            
            CLIENT_CFG="$GENERATED_CONFIGS_DIR/bailian_clients_sf${sf}.toml"
        
            if [ ! -f "$CLIENT_CFG" ]; then
                echo "Error: Client config file not found for scaling factor $sf"
                continue
            fi
        
            for policy in ${POLICIES[@]}; do
                echo "Running experiment for policy: $policy (Run ${run_id})"
                
                max_retries=5
                retry_count=0
                succeeded=false
                
                while [ $retry_count -lt $max_retries ]; do
                    echo "Attempt $(($retry_count + 1))/$max_retries for batch size $bs, scaling factor '$sf', policy '$policy', run r${run_id}..."
                    
                    TIMESTAMP=$(date +%Y%m%d%H%M%S)
                    OUTPUT_DIR="$SF_DIR/${TIMESTAMP}_${policy}"
                    REMOTE_OUTPUT_DIR="$REMOTE_SF_DIR/${TIMESTAMP}_${policy}"
                    mkdir -p "$OUTPUT_DIR"
                    
                    if "$SCRIPT_DIR/bailian_dash.sh" \
                        --output-dir "$OUTPUT_DIR" \
                        --remote-output-dir "$REMOTE_OUTPUT_DIR" \
                        "$BACKEND_CFG" \
                        "$ROUTER_CFG" \
                        "$CLIENT_CFG" \
                        "$policy"; then
                        echo "✅ Experiment completed successfully for batch size $bs, scaling factor $sf, policy $policy, run r${run_id}"
                        succeeded=true
                        break
                    else
                        exit_code=$?
                        echo "❌ Experiment failed for batch size $bs, scaling factor $sf, policy $policy, run r${run_id} with exit code $exit_code"
                        retry_count=$((retry_count + 1))
                        if [ $retry_count -lt $max_retries ]; then
                            echo "⏳ Retrying in 15 seconds..."
                            sleep 15
                        fi
                    fi
                done
                
                if [ "$succeeded" = false ]; then
                    echo "💥 Experiment failed after $max_retries attempts for batch size $bs, scaling factor $sf, policy $policy, run r${run_id}"
                fi
            done
            
            # Move to NFS after all policies for this (bs, sf, run_id)
            echo "Moving TMP DIR to NFS for batch size $bs, scaling factor $sf, run r${run_id}"
            if [[ "$SF_DIR" != /tmp/* ]] || [[ ! -d "$SF_DIR" ]]; then
                echo "ERROR: SF_DIR ('$SF_DIR') is not a valid directory under /tmp/" >&2
                exit 1
            fi
            rsync -av "$SF_DIR" "$STORE_OUTPUT_BASE/" && rm -rf "$SF_DIR"
            
            if [[ "$USE_REMOTE" == "True" ]]; then
                if [[ "$REMOTE_SF_DIR" != /tmp/* ]]; then
                    echo "ERROR: REMOTE_SF_DIR ('$REMOTE_SF_DIR') is not under /tmp/" >&2
                    exit 1
                fi
                ssh -p "$SSH_PORT" "$REMOTE_IPS" "
                    if [[ '$REMOTE_SF_DIR' != /tmp/* ]] || [[ ! -d '$REMOTE_SF_DIR' ]]; then
                        echo 'ERROR: Remote REMOTE_SF_DIR is not a valid directory under /tmp/' >&2
                        exit 1
                    fi
                    rsync -av '$REMOTE_SF_DIR' '$STORE_REMOTE_OUTPUT_BASE/' && rm -rf '$REMOTE_SF_DIR'
                "
            fi
        done
    done
done

echo "All experiments completed."

# -----------------------------------------------------------------------------
# Phase 3: Plotting Results
# -----------------------------------------------------------------------------

echo "Phase 3: Generating plots for each batch size, scaling factor, and run ID..."

for bs in ${BATCH_SIZES[@]}; do
    for run_id in {1..3}; do
        TAG="batch${bs}_u0.9_1118_lwl_bs_tuple_r${run_id}"
        
        for sf in ${SCALING_FACTORS[@]}; do
            echo "Generating plots for batch size: $bs, scaling factor: $sf, run: r${run_id}"
            
            SF_DIR="$STORE_OUTPUT_BASE/${sf}_${TAG}"
        
            if [ ! -d "$SF_DIR" ]; then
                echo "Warning: Directory not found for batch size $bs, scaling factor $sf, run r${run_id}"
                continue
            fi
            echo "try to finding at dir ${SF_DIR}"
            
            unset latest_dirs
            declare -A latest_dirs
        
            while IFS= read -r -d '' dir; do
                if [ -f "$dir/client.jsonl" ]; then
                    basename_dir=$(basename "$dir")
                    if [[ $basename_dir =~ ^[0-9]+_([a-zA-Z0-9_-]+)$ ]]; then
                        policy="${BASH_REMATCH[1]}"
                        if [[ -z "${latest_dirs[$policy]}" ]] || [[ "$basename_dir" > "${latest_dirs[$policy]##*/}" ]]; then
                            latest_dirs[$policy]="$dir"
                        fi
                    fi
                fi
            done < <(find "$SF_DIR" -mindepth 1 -maxdepth 1 -type d -print0)
        
            if [ ${#latest_dirs[@]} -eq 0 ]; then
                echo "⚠️  No valid policy directories found in $SF_DIR"
            else
                sorted_policies=()
                for policy in "${!latest_dirs[@]}"; do
                    sorted_policies+=("$policy")
                done
                IFS=$'\n' sorted_policies=($(sort <<<"${sorted_policies[*]}"))
                unset IFS
                
                dir_list=()
                echo "📊 Found latest directories for policies:"
                for policy in "${sorted_policies[@]}"; do
                    dir="${latest_dirs[$policy]}"
                    echo "  - $policy: $dir"
                    dir_list+=("$dir")
                done
                
                TITLE_TAG="${sf}_${TAG}"
                echo "🎨 Generating comparison plot with compare_cdf.py... ${TITLE_TAG}"
                python3 "$SCRIPT_DIR/../../figures/compare_cdf.py" "${dir_list[@]}" --label "${TITLE_TAG}"
                ret=$?
                if [ $ret -eq 0 ]; then
                    echo "✅ Comparison plot generated successfully for run r${run_id}."
                else
                    echo "❌ compare_cdf.py failed with exit code $ret for run r${run_id}"
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

exit 0
