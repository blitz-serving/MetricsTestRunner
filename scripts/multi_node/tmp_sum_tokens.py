import json
import sys

def main(jsonl_path):
    total_input_200 = 0
    total_output_200 = 0
    total_input_timeout = 0
    total_output_timeout = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                continue

            status = record.get("status")
            input_len = record.get("input_length")
            output_len = record.get("output_length")

            # Ensure input/output lengths are strings or ints and convert to int if possible
            try:
                if input_len is not None:
                    input_len = int(input_len)
                else:
                    input_len = 0
                if output_len is not None:
                    output_len = int(output_len)
                else:
                    output_len = 0
            except (ValueError, TypeError):
                print(f"Warning: Invalid input_length or output_length on line {line_num}", file=sys.stderr)
                continue

            if status == "200":
                total_input_200 += input_len
                total_output_200 += output_len
            elif status == "timeout":
                total_input_timeout += input_len
                total_output_timeout += output_len

    print("Status 200:")
    print(f"  Total input_length: {total_input_200}")
    print(f"  Total output_length: {total_output_200}")
    print()
    print("Status timeout:")
    print(f"  Total input_length: {total_input_timeout}")
    print(f"  Total output_length: {total_output_timeout}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input.jsonl>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])