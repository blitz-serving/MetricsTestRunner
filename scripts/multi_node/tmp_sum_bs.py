import json
import sys

def sum_bs_from_log(file_path):
    total_bs = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                total_bs += record.get("bs", 0)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON line: {line[:50]}... ({e})", file=sys.stderr)
    return total_bs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sum_bs.py <logfile.log>", file=sys.stderr)
        sys.exit(1)

    log_file = sys.argv[1]
    total = sum_bs_from_log(log_file)
    print(f"Total bs: {total}")