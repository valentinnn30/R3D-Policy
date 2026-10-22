"""Token-count rule of run_mug_sweep.sh: argv = success(64) success(128) out.json.
Pick the token count whose success is in [0.30, 0.40]; both -> closest to 0.35;
neither -> closest to 0.35 overall (in_range=false); ties -> 64. Prints the count."""
import json
import math
import sys

s = {64: float(sys.argv[1]), 128: float(sys.argv[2])}
s = {g: v for g, v in s.items() if not math.isnan(v)}
if not s:
    raise SystemExit("no 600-epoch success for 64 or 128 tokens")
inside = {g: v for g, v in s.items() if 0.30 <= v <= 0.40}
pool = inside or s
g = min(pool, key=lambda k: (abs(pool[k] - 0.35), k))
json.dump({"success_600_200scenes": {f"4096/{k}": v for k, v in s.items()}, "chosen_num_group": g,
           "in_range": bool(inside),
           "rule": "success in [0.30, 0.40]; both -> closest to 0.35; neither -> closest to 0.35 overall; ties -> 64"},
          open(sys.argv[3], "w"), indent=2)
print(g)
