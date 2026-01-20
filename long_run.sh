BACKEND_CFGS=(
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/throughput_vllm16.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/throughput_vllm32.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/throughput_vllm64.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/throughput_vllm1.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/throughput_vllm1_fanout4.toml"
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/throughput_vllm2.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/throughput_vllm4.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/throughput_vllm8.toml"
)
ROUTER_CFG="/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/vllm_router_new_fullargs.toml"

CLIENT_CFGS=(
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_6factor.toml"
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_single_request.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_5factor.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_7factor.toml"
)
CLIENT_CFG="/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_single_request.toml"
# POLICIES=("bailian-impl-q" "bounded-most-hit-q" "least-wait-token-q" "round-robin-q")
# POLICIES=("join-shortest-q-ttft" "least-wait-token-mul-bs")
POLICIES=("join-shortest-q-ttft, no-decode-prediction")
# POLICIES=("least-wait-token-mul-bs")



for BACKEND_CFG in "${BACKEND_CFGS[@]}"; do
    for POLICY in "${POLICIES[@]}"; do
        ./scripts/single_node/ipads_h20.sh \
            $BACKEND_CFG \
            $ROUTER_CFG \
            $CLIENT_CFG \
            $POLICY
    done
done
