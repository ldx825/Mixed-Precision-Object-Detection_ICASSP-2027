"""Gate B verdict — auto-generates GATE_B_VERDICT.md from final_table.json
(fallback: eval_summary.json).

Criteria (Addendum §14), margins in AP points (0-1 scale):
  STRONG GO : margin >= +0.005 (= +0.5 AP) at >= 2 practical budgets (5.5/5.0/4.5)
  VERY STRONG: margin >= +0.01 (= +1.0 AP) at >= 1 practical budget
  WEAK/STOP : all practical budgets margin <= +0.002 (= +0.2 AP)
  GRAY      : otherwise (0.002 < margin < 0.005, or only one budget >= 0.005)
"""
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
OUT = ROOT / "outputs/GateB"

PRIORS = ["uniform", "F-LBQ-reimpl", "BLOBQ-reimpl-DP", "BLOBQ-reimpl-greedy", "LampQ-style", "static-OUT"]
OURS = "conditional-OUT"
PRACTICAL = ["5.5", "5.0", "4.5"]


def main():
    # Prefer final_table.json (includes refined allocations); fallback to eval_summary.json.
    src = OUT / "final_table.json"
    if not src.exists():
        src = OUT / "eval_summary.json"
    summ = json.loads(src.read_text())
    lines = ["# Gate B Verdict (auto-generated)", "",
             "Same-budget comparison on COCO val2017 5k (box AP), W-only contract, GT-free calibration.",
             f"Source: {src.name}", ""]
    margins = {}
    for b in sorted([k for k in summ], key=lambda x: -float(x)):
        ms = summ[b]
        # priors = every non-conditional, non-oracle entry (incl. fair-contract 3-bit variants)
        prior_aps = [(m, ms[m]["AP"]) for m in ms
                     if not m.startswith("conditional") and not m.startswith("oracle")]
        cond = [(m, ms[m]["AP"]) for m in ms if m.startswith("conditional")]
        ours_name, ours = max(cond, key=lambda x: x[1]) if cond else (None, None)
        best = max(prior_aps, key=lambda x: x[1]) if prior_aps else None
        lines.append(f"## Budget {b}  (ceilings: FP=42.65)")
        tab = ["| method | AP |  |", "|---|---:|---|"]
        for m, v in sorted(ms.items(), key=lambda x: -x[1]["AP"]):
            mark = " <-- OURS" if m.startswith("conditional") else ""
            tab.append(f"| {m}{mark} | {v['AP']:.4f} |  |")
        lines += tab
        if best and ours is not None:
            margin = ours - best[1]
            margins[b] = margin
            lines.append(f"- strongest prior: **{best[0]} @ {best[1]:.4f}**")
            lines.append(f"- ours: **{ours:.4f}** ({ours_name})")
            lines.append(f"- **margin = {margin:+.4f} AP**")
        elif ours is None:
            lines.append("- (no conditional allocation evaluated yet)")
        lines.append("")

    practical_margins = {b: margins.get(b) for b in PRACTICAL if margins.get(b) is not None}
    go_budgets = [b for b, m in practical_margins.items() if m >= 0.005]
    very_strong = any(m >= 0.01 for m in practical_margins.values())
    all_weak = practical_margins and all(m <= 0.002 for m in practical_margins.values())

    if very_strong and len(go_budgets) >= 1:
        verdict = "**GO (very strong)** — >= +1.0 AP at >= 1 practical budget"
    elif len(go_budgets) >= 2:
        verdict = "**GO (strong)** — >= +0.5 AP at >= 2 practical budgets"
    elif all_weak:
        verdict = "**STOP** — all practical budgets <= +0.2 AP (Addendum §14)"
    else:
        verdict = "**GRAY** — partial evidence; narrow the claim, add budgets/seeds/backbone before final call"

    lines += ["## Verdict", "", f"```\n{verdict}\n```", "",
              f"practical margins (5.5/5.0/4.5): {json.dumps(practical_margins, default=str)}", "",
              "## Caveats (must appear in the paper table)",
              "",
              "1. W-only contract (priors' published detector numbers are W+A); W+A alignment variant runs separately.",
              "2. Calibration: 64 unlabeled images (seed0); cross-seed ranking noise ~0.0016 score_drop units at n=64.",
              "3. 6.0-budget ceiling: FP=42.65 vs uniform-6=42.13 -> margins are structurally compressed there.",
              "4. All method labels: F-LBQ-reimpl / BLOBQ-reimpl (stand-in objective) / LampQ-style (stand-in) — not official runs.",
              "5. Oracle-approx is analysis-only (not a deployable method)."]
    (OUT / "GATE_B_VERDICT.md").write_text("\n".join(lines))
    print("written outputs/GateB/GATE_B_VERDICT.md")
    print(verdict)


if __name__ == "__main__":
    main()
