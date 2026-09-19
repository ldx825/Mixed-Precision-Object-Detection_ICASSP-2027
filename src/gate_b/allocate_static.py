"""Gate B — static allocation policies (CPU; consumes measured signals).

Methods:
  uniform        : all blocks at floor(budget)
  F-LBQ-reimpl   : feature-MSE curves (measured) + gain/cost greedy upgrades
  BLOBQ-reimpl   : output-damage curves (measured) + gain/cost greedy AND exact additive DP
  LampQ-style    : perturbation-sensitivity scalar x shared shape, round-robin upgrades
  static-OUT     : Delta(g: 4->8) from exp40 first step, round-robin upgrades
Budgets: 6.0 / 5.5 / 5.0 / 4.5 / 4.0 (parameter-weighted avg bit over the 12 blocks).
Output: outputs/GateB/allocations_static.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
SIG = json.loads((ROOT / "outputs/GateB_signals/signals.json").read_text())
COUNTS = {g: int(v) for g, v in SIG["counts"].items()}
BLOCKS = sorted(COUNTS)
BITS = [4, 5, 6, 7, 8]
TOTAL = sum(COUNTS.values())
BUDGETS = [6.0, 5.5, 5.0, 4.5, 4.0]


def avg_bit(b):
    return sum(COUNTS[g] * b[g] for g in BLOCKS) / TOTAL


def curve_feat(g):
    return {b: float(SIG["records"][f"{g}@{b}"]["feat"]["rel_mse_all"]) for b in BITS}


def curve_out(g):
    return {b: float(SIG["records"][f"{g}@{b}"]["churn"]["score_drop"]) for b in BITS}


def pert_scalar(g):
    return float(np.mean([SIG["records"][f"{g}~{t}"]["churn"]["score_drop"] for t in range(SIG["pert_trials"])]))


def budget_ok(b, B):
    return avg_bit(b) <= B + 1e-9


def greedy_curves(curves, B):
    b = {g: 4 for g in BLOCKS}
    while True:
        best = None
        for g in BLOCKS:
            if b[g] >= 8:
                continue
            cand = dict(b)
            cand[g] += 1
            if not budget_ok(cand, B):
                continue
            gain = curves[g][b[g]] - curves[g][b[g] + 1]
            if gain <= 0:
                continue
            ratio = gain / COUNTS[g]
            if best is None or ratio > best[1]:
                best = (g, ratio)
        if best is None:
            break
        b[best[0]] += 1
    return b


def dp_alloc(curves, B, nslots=4000):
    """Exact additive DP on discretized cost; returns (bits, modeled loss)."""
    step = TOTAL / nslots
    R = int(round((B - 4.0) * TOTAL / step))
    INF = 1e18
    dp = np.full(R + 1, INF)
    dp[0] = 0.0
    choices = []
    for g in BLOCKS:
        ndp = np.full(R + 1, INF)
        ch = np.zeros(R + 1, dtype=np.int8)
        for k, bits in enumerate(range(4, 9)):
            cslot = 0 if k == 0 else int(round(k * COUNTS[g] / step))
            if cslot > R and k > 0:
                continue
            loss = 0.0 if k == 0 else curves[g][bits]
            cand = np.full(R + 1, INF)
            cand[cslot:] = dp[: R + 1 - cslot] + loss
            upd = cand < ndp
            ndp[upd] = cand[upd]
            ch[upd] = k
        choices.append(ch)
        dp = ndp
    j = int(np.argmin(dp))
    best_loss = float(dp[j])
    b = {}
    for gi in range(len(BLOCKS) - 1, -1, -1):
        g = BLOCKS[gi]
        k = int(choices[gi][j])
        b[g] = 4 + k
        if k > 0:
            j -= int(round(k * COUNTS[g] / step))
            j = max(0, j)
    # safety: if discretization overshot, downgrade largest-block level
    while not budget_ok(b, B):
        g = max(BLOCKS, key=lambda x: COUNTS[x] if b[x] > 4 else -1)
        b[g] -= 1
    return b, best_loss


def round_robin(order, B):
    b = {g: 4 for g in BLOCKS}
    progress = True
    while progress:
        progress = False
        for g in order:
            if b[g] < 8:
                cand = dict(b)
                cand[g] += 1
                if budget_ok(cand, B):
                    b[g] += 1
                    progress = True
    return b


def main():
    out = {}
    # shared output-shape increments (mean over blocks of measured out-curves)
    out_curves = {g: curve_out(g) for g in BLOCKS}

    # signals
    feat_curves = {g: curve_feat(g) for g in BLOCKS}
    lamq_order = sorted(BLOCKS, key=lambda g: -pert_scalar(g))  # most sensitive first
    so_sig_file = ROOT / "outputs/GateB/static_out_signal.json"
    so_order = None
    if so_sig_file.exists():
        so = json.loads(so_sig_file.read_text())
        so_order = sorted(BLOCKS, key=lambda g: -so[g]["delta_4to8"])

    for B in BUDGETS:
        bb = f"{B:.1f}"
        out[bb] = {}
        u = {g: min(8, int(np.floor(B))) for g in BLOCKS}
        out[bb]["uniform"] = {"bits": u, "avg_bit": avg_bit(u)}

        b = greedy_curves(feat_curves, B)
        out[bb]["F-LBQ-reimpl"] = {"bits": b, "avg_bit": avg_bit(b)}

        b = greedy_curves(out_curves, B)
        out[bb]["BLOBQ-reimpl-greedy"] = {"bits": b, "avg_bit": avg_bit(b)}

        b, loss = dp_alloc(out_curves, B)
        out[bb]["BLOBQ-reimpl-DP"] = {"bits": b, "avg_bit": avg_bit(b), "modeled_loss": loss}

        b = round_robin(lamq_order, B)
        out[bb]["LampQ-style"] = {"bits": b, "avg_bit": avg_bit(b)}

        if so_order is not None:
            b = round_robin(so_order, B)
            out[bb]["static-OUT"] = {"bits": b, "avg_bit": avg_bit(b)}

    (ROOT / "outputs/GateB").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/GateB/allocations_static.json").write_text(json.dumps(out, indent=2))
    print("ALLOCATIONS (static) written")
    for B in BUDGETS:
        print(f"\n-- budget {B} --")
        for m, v in out[f"{B:.1f}"].items():
            nz = {g: b for g, b in v["bits"].items() if b != 4}
            print(f"  {m:20s} avg={v['avg_bit']:.3f}  upgrades={nz}")
    print("\nlamq order:", lamq_order)
    if so_order:
        print("static-OUT order:", so_order)


if __name__ == "__main__":
    main()
