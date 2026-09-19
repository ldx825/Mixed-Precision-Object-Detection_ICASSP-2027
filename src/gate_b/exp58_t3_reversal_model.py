"""exp58 — T3 verification: interaction-conditioned reversal floor.

Computes, from the measured 2nd-order tables (Exp30 pairs.csv + centered
interaction matrix):
  1. Exact |S|=1 reversal rate using ONLY 2nd-order data (U_ij, U_i)
     -> validates that pairwise interactions fully explain Exp31's measurement.
  2. Model formula R = 1/2 P(|X| > |D|) with X = I_aj - I_bj, D = U_a - U_b.
  3. R(rho) design curve by scaling X; finds rho* for R = 5%.

Outputs: outputs/GateB/t3_reversal_model.json (+ t3_rho_curve.csv)
"""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
PAIRS = ROOT / "outputs/Exp30_pair_interaction/pairs.csv"
IMAT = ROOT / "outputs/Exp30_pair_interaction/interaction_matrix.npy"
OUT = ROOT / "outputs/GateB"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rows = list(csv.DictReader(open(PAIRS)))
    blocks = sorted(set(r["i"] for r in rows) | set(r["j"] for r in rows))
    idx = {b: i for i, b in enumerate(blocks)}
    n = len(blocks)
    U2 = np.zeros((n, n))
    U1 = np.zeros(n)
    for r in rows:
        i, j = idx[r["i"]], idx[r["j"]]
        U2[i, j] = U2[j, i] = float(r["U_ij"])
        U1[i] = float(r["U_i"])
        U1[j] = float(r["U_j"])
    I_cent = np.load(IMAT)

    # 1) exact |S|=1 enumeration (2nd-order data only)
    flips = 0
    total = 0
    for a, b, j in itertools.permutations(range(n), 3):
        if a > b:
            continue
        d0 = U1[a] - U1[b]
        d1 = U2[a, j] - U2[b, j]
        if abs(d0) < 1e-12 or abs(d1) < 1e-12:
            continue
        total += 1
        if np.sign(d0) != np.sign(d1):
            flips += 1
    R_enum = flips / total

    # 2) model formula
    Ds, Xs = [], []
    for a, b, j in itertools.permutations(range(n), 3):
        if a > b:
            continue
        Ds.append(U1[a] - U1[b])
        Xs.append(I_cent[a, j] - I_cent[b, j])
    Ds = np.array(Ds)
    Xs = np.array(Xs)
    rho0 = float(np.abs(Xs).mean() / np.abs(Ds).mean())
    R_model = float(0.5 * (np.abs(Xs) > np.abs(Ds)).mean())

    # 3) R(rho) curve
    curve = []
    for t in np.linspace(0, 1.5, 31):
        R = float(0.5 * (np.abs(t * Xs) > np.abs(Ds)).mean())
        curve.append({"rho": float(t * rho0), "R": R})
    # rho* for R=5%
    ts = np.linspace(0, 1, 201)
    Rs = [0.5 * (np.abs(t * Xs) > np.abs(Ds)).mean() for t in ts]
    rho5 = None
    for t, R in zip(ts, Rs):
        if R >= 0.05:
            rho5 = float(t * rho0)
            break

    rec = {"n_pairs": n * (n - 1),
           "R_enum_S1": R_enum, "flips": flips, "total": total,
           "R_model": R_model, "rho_mean_ratio": rho0,
           "rho_for_R5pct": rho5,
           "exp31_L1_measured": 0.22575757575757577}
    (OUT / "t3_reversal_model.json").write_text(json.dumps(rec, indent=2))
    with open(OUT / "t3_rho_curve.csv", "w") as f:
        f.write("rho,R\n")
        for p in curve:
            f.write(f"{p['rho']:.4f},{p['R']:.4f}\n")
    print(json.dumps(rec, indent=2))
    print("EXP58 DONE")


if __name__ == "__main__":
    main()
