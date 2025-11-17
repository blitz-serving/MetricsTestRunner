import pandas as pd

def process_csv(input_path, output_path):
    # 读取 CSV
    df = pd.read_csv(input_path)

    # 1. 对缺失的 batch_size 进行插值
    # 先将 batch_size 设为索引
    df = df.set_index("batch_size").sort_index()

    # 生成完整 batch_size 范围（从最小到最大）
    full_index = range(df.index.min(), df.index.max() + 1)

    # 重新索引，缺失值为 NaN
    df = df.reindex(full_index)

    # 对 prediction 进行线性插值。根据你的描述，插值方式是前后平均——
    # 线性插值对这种等间隔情况正好等价。
    df["predictions"] = df["predictions"].interpolate(method="linear")

    # 2. 将 batch_size 扩展到 200，并将新增的 prediction 设置为最大 batch_size 的 prediction
    max_bs = df.index.max()
    max_pred = df.loc[max_bs, "predictions"]

    # 扩展到 200
    for bs in range(max_bs + 1, 251):
        df.loc[bs] = max_pred

    # 保存结果
    df = df.reset_index().rename(columns={"index": "batch_size"})
    df.to_csv(output_path, index=False)
    print(f"处理完成，结果已保存到: {output_path}")


# 示例调用
if __name__ == "__main__":
    process_csv("/nvme/zkx/MetricsTestRunner/before_execute_avg.csv", "processed_before.csv")
    process_csv("/nvme/zkx/MetricsTestRunner/sampler.csv", "processed_after.csv")