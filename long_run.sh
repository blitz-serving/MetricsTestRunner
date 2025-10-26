BACKEND_CFG="/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/launch_vllm_dp8.toml"
ROUTER_CFG="/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/vllm_router.toml"
CLIENT_CFG="/nvme/zkx/MetricsTestRunner/config/ipads-h20-1/new_clients.toml"

POLICIES=("least-work-q", "round-robin-q", "join-shortest-q")

WORK_DIR="/nvme/zkx/blitz-infer-pack"
VENV_PATH="/nvme/zkx/modified-vllm/myenv/bin/"


# For automatic end-to-end testing of all policies
FOR POLICY IN "${POLICIES[@]}"; do
    ${WORK_DIR}/MetricsTestRunner/scripts/single_node/ipads_h20.sh \
        $BACKEND_CFG \
        $ROUTER_CFG \
        $CLIENT_CFG \
        $POLICY
done


# For components start-up
FOR PART IN "backend" "router" "clients"; do
${WORK_DIR}/MetricsTestRunner/scripts/single_node/human_in_the_loop.sh \
    $BACKEND_CFG \
    $ROUTER_CFG \
    $CLIENT_CFG \
    $POLICY \
    $PART
done 