import re, sys

log = sys.argv[1] if len(sys.argv) > 1 else "importtime.log"
rows = []
for line in open(log, encoding="utf-8", errors="ignore"):
    m = re.match(r"import time:\s+(\d+) \|\s+(\d+) \|( *)(\S+)", line)
    if m:
        cumulative_us, indent, name = int(m.group(2)), len(m.group(3)), m.group(4)
        if indent == 1:  # one space = a top-level import (nested imports are indented further)
            rows.append((cumulative_us, name))
print("Slowest top-level imports (cumulative seconds):")
for cumulative_us, name in sorted(rows, reverse=True)[:15]:
    print(f"{cumulative_us / 1e6:7.2f}s  {name}")
print(f"\nSum of all top-level imports: {sum(r[0] for r in rows) / 1e6:.1f}s")
