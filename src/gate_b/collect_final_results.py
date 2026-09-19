"""collect_final_results.py — merge ALL result sources into one final table.

Sources:
  - outputs/GateB/eval_summary.json                (baselines + conditional + oracle, 5k)
  - outputs/GateB/power_refined/*.json             (exp50 power refinements)
  - outputs/GateB/power_refined2/*.json            (exp52 xl, exp53 shgreedy)
Output: outputs/GateB/FINAL_TABLE.md + final_table.json
"""
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
OUT = ROOT / "outputs/GateB"

PRIORS = ["uniform", "F-LBQ-reimpl", "BLOBQ-reimpl-DP", "BLOBQ-reimpl-greedy", "LampQ-style", "static-OUT"]


def main():
    table = {}  # budget -> {label: {"AP":.., "avg_bit":.., "src":..}}

    def put(b, label, ap, avg, src):
        key = f"{b}"
        table.setdefault(key, {})
        if label not in table[key] or ap > table[key][label]["AP"]:
            table[key][label] = {"AP": ap, "avg_bit": avg, "src": src}

    s = json.loads((OUT / "eval_summary.json").read_text())
    for b, ms in s.items():
        for m, v in ms.items():
            put(b, m, v["AP"], None, "eval_summary")

    for d, src in [(OUT / "power_refined", "exp50"), (OUT / "power_refined2", "exp52-55")]:
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("bak"):
                continue
            rec = json.loads(f.read_text())
            # tag from filename prefix (first token before _)
            b = f.name.split("_")[0]
            if "AP" not in rec:
                continue
            if f.stem.endswith("_v3"):
                label = "conditional-v3"
            elif f.stem.endswith("_pair"):
                label = "conditional-pair"
            elif f.stem.endswith("_xl"):
                label = "conditional-xl"
            elif f.stem.endswith("_shgreedy"):
                label = "conditional-shgreedy"
            elif "power" in f.stem:
                label = "conditional-power"
            else:
                label = "conditional-refined"
            put(b, label, rec["AP"], rec.get("avg_bit"), f"{src}:{f.name}")

    lines = ["# FINAL RESULTS TABLE (5k bbox AP, all GT-free)", ""]
    jout = {}
    for b in sorted(table, key=lambda x: -float(x)):
        ms = table[b]
        priors = [(m, v) for m, v in ms.items() if m in PRIORS]
        ours = [(m, v) for m, v in ms.items() if "conditional" in m]
        best_prior = max(priors, key=lambda x: x[1]["AP"]) if priors else None
        best_ours = max(ours, key=lambda x: x[1]["AP"]) if ours else None
        lines.append(f"## Budget {b}")
        lines.append("| method | AP | avg_bit | src |")
        lines.append("|---|---:|---:|---|")
        for m, v in sorted(ms.items(), key=lambda x: -x[1]["AP"]):
            star = "  ← OURS" if "conditional" in m else ("  ← best prior" if best_prior and m == best_prior[0] else "")
            lines.append(f"| {m}{star} | {v['AP']:.4f} | {v['avg_bit'] if v['avg_bit'] else '—'} | {v['src']} |")
        if best_prior and best_ours:
            margin = best_ours[1]["AP"] - best_prior[1]["AP"]
            lines.append(f"**margin (ours-best − best-prior): {margin:+.4f}**  "
                         f"({best_ours[0]} vs {best_prior[0]})")
        lines.append("")
        jout[b] = ms
    (OUT / "FINAL_TABLE.md").write_text("\n".join(lines))
    (OUT / "final_table.json").write_text(json.dumps(jout, indent=2))
    print("\n".join(lines))
    print("written FINAL_TABLE.md")


if __name__ == "__main__":
    main()
