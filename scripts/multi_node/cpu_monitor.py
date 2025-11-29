#!/usr/bin/env python3
import psutil
import time
import datetime
import argparse
import sys

def monitor_cpu(output_file: str, interval: int = 5):
    print(f"开始监测 CPU 利用率，每 {interval} 秒记录一次，输出文件：{output_file}")
    try:
        with open(output_file, 'a', buffering=1) as f:  # 行缓冲确保实时写入
            while True:
                cpu_percent = psutil.cpu_percent(interval=None)  # 非阻塞采样
                timestamp = datetime.datetime.now().isoformat()
                line = f"{timestamp},{cpu_percent:.2f}\n"
                f.write(line)
                print(f"{timestamp} - CPU: {cpu_percent:.2f}%")
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n监测已停止。")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="监测 CPU 利用率并写入文件")
    parser.add_argument("-o", "--output", default="cpu_usage.log", help="输出文件路径（默认：cpu_usage.log）")
    parser.add_argument("-i", "--interval", type=int, default=5, help="采样间隔（秒，默认：5）")
    args = parser.parse_args()

    monitor_cpu(args.output, args.interval)