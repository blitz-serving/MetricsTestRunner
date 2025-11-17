BACKEND_CFG="/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/launch_vllm_dp8.toml"
ROUTER_CFG="/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/vllm_router.toml"

CLIENT_CFGS=(
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_6factor.toml"
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_6factor.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_5factor.toml"
    # "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_7factor.toml"
)
# POLICIES=("bailian-impl-q" "bounded-most-hit-q" "least-wait-token-q" "round-robin-q")
# POLICIES=("bailian-impl-q" "round-robin-q")
POLICIES=("round-robin-q")

# WORK_DIR="/nvme/zkx/blitz-infer-pack"
# VENV_PATH="/nvme/zkx/modified-vllm/myenv/bin/"

# ./scripts/single_node/ipads_h20.sh \
#     $BACKEND_CFG \
#     $ROUTER_CFG \
#     $CLIENT_CFG \
#     "round-robin-q"


# For automatic end-to-end testing of all policies
# for CLIENT_CFG in "${CLIENT_CFGS[@]}"; do
#     export CLIENT_CFG
#     for POLICY in "${POLICIES[@]}"; do
#         ./scripts/single_node/ipads_h20.sh \
#             $BACKEND_CFG \
#             $ROUTER_CFG \
#             $CLIENT_CFG \
#             $POLICY
#     done
# done

# ./scripts/single_node/ipads_h20_llama.sh \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/dense_6simulator.toml" \
#     $ROUTER_CFG \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_azure4_2.toml" \
#     "round-robin-q"

./scripts/single_node/ipads_h20.sh \
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/launch_vllm_dp4_flashinfer.toml" \
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/vllm_router_new.toml" \
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_single_request.toml" \
    "join-shortest-q-ttft" # "round-robin-q"

./scripts/single_node/ipads_llama.sh \
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/launch_vllm_1.toml" \
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/vllm_router_new.toml" \
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_single_request.toml" \
    "round-robin-q"
    

# ./scripts/single_node/ipads_h20_llama.sh \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/launch_vllm_dp8.toml" \
#     $ROUTER_CFG \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_azure4_2.toml" \
#     "round-robin-q"

# ./scripts/single_node/human_in_the_loop.sh "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/launch_vllm_1.toml" \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/vllm_router.toml" \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_single_request.toml" \
#     "round-robin-q"
#     --components "backend"


# ./scripts/single_node/human_in_the_loop.sh "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/launch_vllm_1.toml" \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/vllm_router_new.toml" \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_single_request.toml" \
#     "join-shortest-q-ttft"
#     --components "backend"

# ./scripts/single_node/human_in_the_loop.sh "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/launch_vllm_1.toml" \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/vllm_router_new.toml" \
#     "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_single_request.toml" \
#     "join-shortest-q-ttft"
#     --components "backend"
# # For components start-up
# FOR PART IN "backend" "router" "clients"; do
# ${WORK_DIR}/MetricsTestRunner/scripts/single_node/human_in_the_loop.sh \
#     $BACKEND_CFG \
#     $ROUTER_CFG \
#     $CLIENT_CFG \
#     $POLICY \
#     $PART
# done 