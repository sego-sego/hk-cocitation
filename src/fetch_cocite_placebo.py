"""Placebo for the co-citation design: topical breadth of j.

For pairs (i, j) already in cocite.csv, pick a PLOS ONE target i' from the
same field as i that j does NOT cite, and count works citing both j and i'.
This is the background co-citation rate of j with a comparable paper it did
not pass on.  If it also falls with k_j, the k-dependence in cocite.csv
reflects the breadth of long-list works, not transfer through the reference.

Usage: python fetch_cocite_placebo.py --data data --per-bin 1000
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import gzip
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import fetch_openalex as F
from fetch_cocite import K_EDGES, RateLimiter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--per-bin", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--match", default="neighbor", choices=["neighbor", "field"],
                    help="neighbor: the PLOS ONE paper most often co-cited with i elsewhere "
                         "(topically closest), not cited by j; field: random paper of i's field")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    field_of, by_field = {}, {}
    for l in gzip.open(os.path.join(a.data, "targets.jsonl.gz"), "rt"):
        r = json.loads(l)
        f = r.get("field") or ""
        field_of[r["id"]] = f
        by_field.setdefault(f, []).append(r["id"])
    cites_of = {}
    from collections import Counter, defaultdict
    nb = defaultdict(Counter)
    for fn in sorted(glob.glob(os.path.join(a.data, "citing", "*.jsonl.gz"))):
        for l in gzip.open(fn, "rt"):
            r = json.loads(l)
            c = list(set(r.get("cites") or []))
            cites_of[r["id"]] = set(c)
            if a.match == "neighbor" and 1 < len(c) <= 50:
                for x in c:
                    for y in c:
                        if x != y:
                            nb[x][y] += 1
    rows = list(csv.DictReader(open(os.path.join(a.data, "cocite.csv"))))
    rng.shuffle(rows)
    take, cnt = [], [0] * (len(K_EDGES) - 1)
    for r in rows:
        b = bisect.bisect_right(K_EDGES, int(r["k"])) - 1
        if cnt[b] >= a.per_bin:
            continue
        cited = cites_of.get(r["j"], set())
        ip = None
        if a.match == "neighbor":
            # topically closest paper: most co-cited with i by works other than j
            for cand, _ in nb[r["i"]].most_common(30):
                if cand not in cited and cand != r["i"]:
                    ip = cand
                    break
        else:
            pool = by_field.get(field_of.get(r["i"], ""), [])
            for _ in range(20):
                c = rng.choice(pool) if pool else None
                if c and c not in cited and c != r["i"]:
                    ip = c
                    break
        if ip is None:
            continue
        cnt[b] += 1
        take.append((r, ip))
    out = os.path.join(a.data, "cocite_placebo.csv" if a.match == "field" else "cocite_placebo_nb.csv")
    done = set()
    if os.path.exists(out):
        done = {(x["i"], x["j"]) for x in csv.DictReader(open(out))}
    todo = [(r, ip) for r, ip in take if (r["i"], r["j"]) not in done]
    print(f"placebo pairs: {len(take)} (per k-bin {cnt}); {len(done)} done, {len(todo)} to go", flush=True)
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["i", "j", "k", "date", "type", "field", "n_j", "source", "y", "i_placebo", "y_placebo"])
    lock, lim, t0, st = threading.Lock(), RateLimiter(a.rate), time.time(), dict(n=0, err=0)

    def one(item):
        r, ip = item
        lim.wait()
        try:
            yp = F.get("/works", {"filter": f"cites:{r['j']},cites:{ip}", "per-page": 1, "select": "id"})["meta"]["count"]
        except Exception:  # noqa: BLE001
            with lock:
                st["err"] += 1
            return
        with lock:
            w.writerow([r["i"], r["j"], r["k"], r["date"], r["type"], r["field"], r["n_j"], r["source"], r["y"], ip, yp])
            st["n"] += 1
            if st["n"] % 50 == 0:
                fh.flush()
            el = time.time() - t0
            F.status(f"placebo: {st['n']}/{len(todo)} | errors {st['err']} | elapsed {F.fmt_time(el)}, "
                     f"~{F.fmt_time(el / st['n'] * (len(todo) - st['n']))} left", force=(st["n"] % 20 == 0))

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, todo))
    fh.close()
    print(f"\ndone: {st['n']} written to {out}; {st['err']} failed (rerun to retry)")


if __name__ == "__main__":
    main()
