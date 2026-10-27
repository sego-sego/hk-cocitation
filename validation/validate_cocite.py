"""End-to-end validation of the co-citation pipeline on synthetic data.

Generates a synthetic cohort in the OpenAlex format (validation/synth.py),
replaces the OpenAlex API by a mock that draws co-citation counts
    Y_ij ~ Poisson(n_j * a_i * (k_j / 35)^(-gamma_true)),
runs src/fetch_cocite.py and src/cocite_analyze.py unchanged, and prints the
estimate.  Expected: gamma_hat close to gamma_true (paper, Appendix B).

Usage (from the repository root):
    python validation/validate_cocite.py --gamma 0.6 --out /tmp/hk_validation
"""
import argparse
import glob
import gzip
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--out", default="/tmp/hk_validation")
    ap.add_argument("--per-bin", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    data = os.path.join(a.out, "synth")
    if not os.path.exists(os.path.join(data, "targets.jsonl.gz")):
        subprocess.run([sys.executable, os.path.join(ROOT, "validation", "synth.py"), "--out", data,
                        "--gamma", "0.6", "--level", "0.008"], check=True)
    os.environ["OPENALEX_API_KEY"] = "mock"
    import fetch_openalex as F
    import fetch_cocite as FC

    rng = np.random.default_rng(a.seed)
    meta = {}
    for fn in glob.glob(os.path.join(data, "citing", "*.jsonl.gz")):
        for line in gzip.open(fn, "rt"):
            r = json.loads(line)
            meta[r["id"]] = (r["k"], r["cited_by"])
    level = {}

    def mock_get(path, params, retries=12):
        j, i = [x.split(":")[1] for x in params["filter"].split(",")]
        ai = level.setdefault(i, np.exp(rng.normal(-3, 0.8)))
        k, nj = meta[j]
        return {"results": [], "meta": {"count": int(rng.poisson(nj * ai * (k / 35) ** (-a.gamma)))}}

    F.get = mock_get
    out_csv = os.path.join(data, "cocite.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    sys.argv = ["fetch_cocite.py", "--data", data, "--per-bin", str(a.per_bin), "--rate", "1000"]
    FC.main()
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "cocite_analyze.py"), "--data", data,
                    "--out", os.path.join(a.out, "results")], check=True)
    print(f"true gamma = {a.gamma}")


if __name__ == "__main__":
    main()
