#!/bin/bash



run_all_policies_with_retry() {
    # "round-robin-q"
    # "bounded-most-hit-q"
    # "least-wait-token-q"
    # "bailian-impl-q"
    local policies=(
        "round-robin-q"
        "bounded-most-hit-q"
        "least-wait-token-q"
        "bailian-impl-q"
        "join-shortest-q-weight"
        "join-shortest-q-tuple"
    )

    local backend_cfg="../../config/dash-h20-1/launch_vllm_16instances.toml"
    local router_cfg="../../config/dash-h20-1/vllm_router.toml"
    local client_cfg="../../config/dash-h20-1/bailian_clients.toml"

    local max_retries=3
    local overall_success=true

    for policy in "${policies[@]}"; do
        local retry_count=0
        local succeeded=false

        echo "================================================================"
        echo "🚀 Starting experiment for policy: $policy"
        echo "================================================================"

        while [ $retry_count -lt $max_retries ]; do
            echo "Attempt $(($retry_count + 1))/$max_retries for policy '$policy'..."

            if ./bailian_dash.sh "$backend_cfg" "$router_cfg" "$client_cfg" "$policy"; then
                echo "✅ Policy '$policy' completed successfully."
                succeeded=true
                break
            else
                local exit_code=$?
                echo "❌ Policy '$policy' failed with exit code $exit_code"
                retry_count=$((retry_count + 1))
                if [ $retry_count -lt $max_retries ]; then
                    echo "⏳ Retrying in 15 seconds..."
                    sleep 15
                fi
            fi
        done

        if [ "$succeeded" = false ]; then
            echo "💥 Policy '$policy' failed after $max_retries attempts."
            overall_success=false
        fi

        sleep 5
    done

    echo "================================================================"
    if [ "$overall_success" = true ]; then
        echo "🎉 All policies completed successfully!"
        return 0
    else
        echo "⚠️  Some policies failed. Check logs for details."
        return 1
    fi
}


plot_latest_policy_comparison() {
    local log_root="/mnt/debugger/hjb/node1/lmmetric-logs"
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local compare_script="$script_dir/../../figures/compare_cdf.py"

    # 检查 compare_cdf.py 是否存在
    if [ ! -f "$compare_script" ]; then
        echo "❌ Error: compare_cdf.py not found at $compare_script"
        return 1
    fi

    # 确保日志目录存在
    if [ ! -d "$log_root" ]; then
        echo "❌ Error: log directory not found: $log_root"
        return 1
    fi

    # 声明关联数组：policy -> latest_dir
    declare -A latest_dirs

    # 遍历所有子目录
    while IFS= read -r -d '' dir; do
        basename_dir=$(basename "$dir")
        # 匹配格式：数字_策略名（至少一个数字 + 下划线 + 非空策略名）
        if [[ $basename_dir =~ ^[0-9]+_([a-zA-Z0-9_-]+)$ ]]; then
            policy="${BASH_REMATCH[1]}"
            # 只处理我们关心的策略（可选，也可不限制）
            case "$policy" in
                round-robin-q|join-shortest-q|bounded-most-hit-q|least-wait-token-q|bailian-impl-q|join-shortest-q-weight|join-shortest-q-tuple)
                    # 如果该策略尚未记录，或当前时间戳更大，则更新
                    if [[ -z "${latest_dirs[$policy]}" ]] || [[ "$basename_dir" > "${latest_dirs[$policy]##*/}" ]]; then
                        latest_dirs[$policy]="$dir"
                    fi
                    ;;
            esac
        fi
    done < <(find "$log_root" -mindepth 1 -maxdepth 1 -type d -print0)

    # 检查是否找到任何有效目录
    if [ ${#latest_dirs[@]} -eq 0 ]; then
        echo "⚠️  No valid policy directories found in $log_root"
        return 1
    fi

    # 按策略名排序（可选，使顺序一致）
    sorted_policies=()
    for policy in "${!latest_dirs[@]}"; do
        sorted_policies+=("$policy")
    done
    IFS=$'\n' sorted_policies=($(sort <<<"${sorted_policies[*]}"))
    unset IFS

    # 构建目录列表
    dir_list=()
    echo "📊 Found latest directories for policies:"
    for policy in "${sorted_policies[@]}"; do
        dir="${latest_dirs[$policy]}"
        echo "  - $policy: $dir"
        dir_list+=("$dir")
    done

    # 调用 compare_cdf.py
    echo "🎨 Generating comparison plot with compare_cdf.py..."
    python3 "$compare_script" "${dir_list[@]}"
    local ret=$?
    if [ $ret -eq 0 ]; then
        echo "✅ Comparison plot generated successfully."
    else
        echo "❌ compare_cdf.py failed with exit code $ret"
    fi
    return $ret
}

run_all_policies_with_retry


plot_latest_policy_comparison
