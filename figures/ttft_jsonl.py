import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot CDF comparison of first_token_time from two directories.")
parser.add_argument("dir_a", type=str, help="Path to the first directory (e.g., LWL)")
parser.add_argument("dir_b", type=str, help="Path to the second directory (e.g., RR)")
args = parser.parse_args()

dir_a = args.dir_a
dir_b = args.dir_b

# Extract label: take the last part of the directory name after splitting by '_'
def extract_label_from_path(path):
    basename = os.path.basename(os.path.normpath(path))
    parts = basename.split('_')
    return parts[-1] if parts else basename

label_a = extract_label_from_path(dir_a)
label_b = extract_label_from_path(dir_b)

# Output plot filename
plot_filename = "ttft.png"
output_plot_a = os.path.join(dir_a, plot_filename)
output_plot_b = os.path.join(dir_b, plot_filename)

# Fields of interest
fields = ["first_token_time"]


def load_all_jsonl(directory, fields):
    data = {field: [] for field in fields}
    jsonl_files = glob.glob(os.path.join(directory, "client.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {directory}")
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    for field in fields:
                        raw_value = obj.get(field, 0)
                        if raw_value == "nil":
                            value = 0.0
                        else:
                            value = float(raw_value)
                        data[field].append(value)
                except Exception as e:
                    print(f"Skip arsing line in {file_path}: {e}")
                    continue
    return data

# Load data from both directories
data_a = load_all_jsonl(dir_a, fields)
data_b = load_all_jsonl(dir_b, fields)

# Plot CDF
plt.figure(figsize=(8, 6))

# Use extracted labels for legend
labels = [(data_a, label_a), (data_b, label_b)]
linestyles = {label_a: "-", label_b: "--"}

for field in fields:
    for dataset, label in labels:
        arr = np.array(dataset[field])
        if len(arr) == 0:
            print(f"No data for {field} in {label}")
            continue
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)

        mean_val = np.mean(arr)
        legend_label = f"{field} ({label}) - mean: {mean_val:.2f} ms"

        plt.plot(sorted_arr, cdf, 
                 linestyle=linestyles[label], 
                 label=legend_label)

plt.xlabel("Value (ms)")
plt.ylabel("CDF")
plt.title(f"CDF Comparison of {label_a} vs {label_b}")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot to both directories
plt.savefig(output_plot_a)
plt.savefig(output_plot_b)
plt.close()

print(f"Plots saved to {output_plot_a} and {output_plot_b}")