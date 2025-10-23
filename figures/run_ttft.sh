#!/bin/bash
cd "$(dirname "$0")" || exit 1

LOGS_DIR="$HOME/lmmetric-logs"

if [ ! -d "$LOGS_DIR" ]; then
    echo "Error: $LOGS_DIR does not exist."
    exit 1
fi

dirs=($(ls -1t "$LOGS_DIR" | head -n 2))

if [ ${#dirs[@]} -lt 2 ]; then
    echo "Error: Less than two subdirectories found in $LOGS_DIR."
    exit 1
fi

dir1="$LOGS_DIR/${dirs[0]}"
dir2="$LOGS_DIR/${dirs[1]}"

echo "Using directories:"
echo "  1st: $dir1"
echo "  2nd: $dir2"

python ttft_jsonl.py "$dir1" "$dir2"