BACKEND_CFG="/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/launch_vllm_dp8.toml"
ROUTER_CFG="/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/vllm_router.toml"

CLIENT_CFGS=(
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_6factor.toml"
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_4factor.toml"
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_5factor.toml"
    "/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients_7factor.toml"
)
POLICIES=("bailian-impl-q" "bounded-most-hit-q" "least-wait-token-q" "round-robin-q")

WORK_DIR="/nvme/zkx/blitz-infer-pack"
VENV_PATH="/nvme/zkx/modified-vllm/myenv/bin/"

# ./scripts/single_node/ipads_h20.sh \
#     $BACKEND_CFG \
#     $ROUTER_CFG \
#     $CLIENT_CFG \
#     "round-robin-q"


# For automatic end-to-end testing of all policies
for CLIENT_CFG in "${CLIENT_CFGS[@]}"; do
    export CLIENT_CFG
    for POLICY in "${POLICIES[@]}"; do
        ./scripts/single_node/ipads_h20.sh \
            $BACKEND_CFG \
            $ROUTER_CFG \
            $CLIENT_CFG \
            $POLICY
    done
done

# # For components start-up
# FOR PART IN "backend" "router" "clients"; do
# ${WORK_DIR}/MetricsTestRunner/scripts/single_node/human_in_the_loop.sh \
#     $BACKEND_CFG \
#     $ROUTER_CFG \
#     $CLIENT_CFG \
#     $POLICY \
#     $PART
# done 