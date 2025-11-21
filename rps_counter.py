import re
from collections import Counter

with open("/nvme/lmetric/logs/lmmetric-logs/20251114074157_round-robin-q/router_v2.log", "r", encoding="utf-8") as f:
    content = f.read()

pattern = re.compile(
    r"(?<!\d)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?!\d)"
)

timestamps = pattern.findall(content)

counter = Counter(timestamps)

for ts in sorted(counter):
    print(f"{ts} -> {counter[ts]} 条")
