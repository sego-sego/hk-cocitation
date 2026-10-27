"""Co-citation design: for a sample of citation pairs (target i, citing work j)
count the later works that cite BOTH j and i.

Interpretation.  A reader of j meets i as one of the k_j references of j.
Y_ij = #(works citing j and i) out of n_j = #(works citing j) measures how
much of j's audience j passes on to i.  If j's attention is split among its
references (fractional regime), Y_ij / n_j falls as 1/k_j; if every reference
is passed on in full (public regime), it does not depend on k_j.

Each pair costs one OpenAlex request (filter cites:j,cites:i, count only).
Pairs are sampled evenly over k-bins.  Resumable; runs several requests in
parallel within the OpenAlex rate limit.

Usage:
    export OPENALEX_API_KEY=...
    python fetch_cocite.py --data data --per-bin 4000
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import fetch_openalex as F

K_EDGES = (1, 16, 26, 36, 51, 76, 121, 10 ** 6)


def load_pairs(data, last_year):
    tgt = {}
    for l in gzip.open(os.path.join(data, "targets.jsonl.gz"), "rt"):
        r = json.loads(l)
        tgt[r["id"]] = set(r.get("authors") or [])
    pairs = []
    for fn in sorted(glob.glob(os.path.join(data, "citing", "*.jsonl.gz"))):
        for l in gzip.open(fn, "rt"):
            r = json.loads(l)
            if not r.get("date") or int(r["date"][:4]) > last_year:
                continue
            k = max(int(r.get("k") or 0), int(r.get("k_list") or 0), 1)
            au = set(r.get("authors") or [])
            for t in r.get("cites") or []:
                if t in tgt and not (au & tgt[t]):
                    pairs.append((t, r["id"], k, r["date"], r.get("type") or "", r.get("field") or "",
                                  r.get("cited_by") or 0, r.get("source") or ""))
    return pairs


class RateLimiter:
    def __init__(self, per_sec):
        self.dt = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next = time.time()

    def wait(self):
        with self.lock:
            now = time.time()
            if self.next > now:
                time.sleep(self.next - now)
            self.next = max(self.next, now) + self.dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--per-bin", type=int, default=4000)
    ap.add_argument("--last-year", type=int, default=2018,
                    help="only citing works published up to this year (time for j to be cited)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rate", type=float, default=8.0, help="requests per second")
    ap.add_argument("--seed", type=int, default=20261025)
    a = ap.parse_args()
    if not os.environ.get("OPENALEX_API_KEY"):
        print("warning: OPENALEX_API_KEY not set; requests may be throttled")

    pairs = load_pairs(a.data, a.last_year)
    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    import bisect
    take, cnt = [], [0] * (len(K_EDGES) - 1)
    for p in pairs:
        b = bisect.bisect_right(K_EDGES, p[2]) - 1
        if cnt[b] < a.per_bin:
            cnt[b] += 1
            take.append(p)
    print(f"{len(pairs)} eligible pairs; sampled {len(take)} (per k-bin: {cnt})", flush=True)

    out = os.path.join(a.data, "cocite.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            done = {(r["i"], r["j"]) for r in csv.DictReader(fh)}
    todo = [p for p in take if (p[0], p[1]) not in done]
    print(f"{len(done)} already done, {len(todo)} to go", flush=True)
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["i", "j", "k", "date", "type", "field", "n_j", "source", "y"])
        fh.flush()
    lock = threading.Lock()
    lim = RateLimiter(a.rate)
    t0 = time.time()
    state = dict(n=0, err=0)

    def one(p):
        i, j = p[0], p[1]
        lim.wait()
        try:
            r = F.get("/works", {"filter": f"cites:{j},cites:{i}", "per-page": 1, "select": "id"})
            y = r["meta"]["count"]
        except Exception as e:  # noqa: BLE001 - log and skip, rerun picks it up
            with lock:
                state["err"] += 1
            return
        with lock:
            w.writerow(list(p) + [y])
            state["n"] += 1
            if state["n"] % 50 == 0:
                fh.flush()
            n = state["n"]
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            F.status(f"co-citations: {n}/{len(todo)} pairs ({100 * n / len(todo):.0f}%) | errors {state['err']} | "
                     f"elapsed {F.fmt_time(el)}, ~{F.fmt_time(eta)} left", force=(n % 20 == 0))

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, todo))
    fh.close()
    print(f"\ndone: {state['n']} pairs written to {out}; {state['err']} failed (rerun to retry)")


if __name__ == "__main__":
    sys.exit(main())
