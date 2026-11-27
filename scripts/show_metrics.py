"""Print a compact summary of a closed-loop run from its metrics.json.

    python scripts/show_metrics.py runs/potts_L16_nbt2x6/metrics.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/loop_default/metrics.json")
    if path.suffix == ".jsonl":
        # live view of an unbounded run: metrics.json is only written on exit
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        m = {"config": {"arch": "?", "q": 3}, "params": 0, "history": rows}
    else:
        m = json.loads(path.read_text(encoding="utf-8"))
    c, h = m["config"], m["history"]
    print(f"{path}")
    print(f"  arch={c.get('arch')} overrides={c.get('arch_overrides')} "
          f"L={c.get('L')} q={c.get('q')} params={m.get('params'):,}")
    print(f"  beta {c.get('beta_start')}->{c.get('beta')}  rounds done={len(h)}  "
          f"optimizer={c.get('optimizer')}  gate={c.get('gate_mode')}")
    total_h = sum(x["seconds"] for x in h) / 3600
    print(f"  total {total_h:.2f} h, {sum(x['accepted'] for x in h)}/{len(h)} promoted")
    if not h:
        return 0
    print()
    print(f"  {'rnd':>4} {'beta':>6} {'ESS':>7} {'loss':>8} {'NLL':>8} {'gate':>7} {'s':>6}")
    step = max(1, len(h) // 25)
    for x in h[::step] + ([h[-1]] if (len(h) - 1) % step else []):
        print(f"  {x['round']:>4} {x['beta']:>6.3f} {x['scan_ess']:>7.4f} "
              f"{x['train_loss_mean']:>8.3f} {-x['score_after']:>8.5f} "
              f"{'kept' if x['accepted'] else 'rolled':>7} {x['seconds']:>6.1f}")
    first, last = -h[0]["score_after"], -h[-1]["score_after"]
    print(f"\n  holdout NLL {first:.5f} -> {last:.5f}   "
          f"(uniform ln(q+1) = {math.log(c.get('q', 3) + 1):.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
