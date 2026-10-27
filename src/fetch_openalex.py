"""Download the raw data for the h(k) analysis from OpenAlex.

Stage 1 (targets): all PLOS ONE research articles of the chosen publication
years, with publication date and author IDs.
Stage 2 (citing works): every work that cites at least one target, with its
publication date, type, reference-list length (referenced_works_count),
the targets it cites, author IDs, source, field and total citations.

Only the fields needed downstream are kept, in compact gzipped JSON lines.
Both stages are resumable: rerunning skips finished batches.

Usage:
    python fetch_openalex.py --out data --years 2010
Optional environment variables:
    OPENALEX_API_KEY   API key, if your OpenAlex account requires one
    OPENALEX_MAILTO    e-mail for the polite pool
Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import gzip
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.openalex.org"
PLOS_ONE_ISSN = "1932-6203"


def short(x):
    return x.rsplit("/", 1)[-1] if x else None


_last_status = [0.0]


def status(msg, force=False):
    """One self-overwriting status line (at most ~2 updates per second)."""
    now = time.time()
    if force or now - _last_status[0] > 0.5:
        sys.stdout.write("\r" + msg[:150].ljust(150))
        sys.stdout.flush()
        _last_status[0] = now


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def get(path, params, retries=12):
    params = dict(params)
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    if os.environ.get("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    url = f"{BASE}{path}?{urllib.parse.urlencode(params, safe=':|,*')}"
    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hk-analysis/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                status(f"HTTP {e.code}, retry {attempt + 1} in {delay:.0f} s", force=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            body = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {e.code} for {url}\n{body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
                http.client.IncompleteRead, http.client.HTTPException, json.JSONDecodeError):
            # dropped or truncated responses: wait and retry the same request
            status(f"connection problem, retry {attempt + 1} in {delay:.0f} s", force=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"giving up on {url}")


def paged(path, params, progress=None):
    """All results of a query.  With an API key: cursor paging (no limit).
    Without a key OpenAlex refuses cursor paging, so fall back to page=1,2,...
    which works for up to 10,000 results; larger queries are rejected with a
    clear message (they need the key)."""
    if os.environ.get("OPENALEX_API_KEY"):
        cursor = "*"
        while cursor:
            page = get(path, {**params, "per-page": 100, "cursor": cursor})
            if progress:
                progress(len(page["results"]), page["meta"].get("count"))
            yield from page["results"]
            cursor = page["meta"].get("next_cursor")
            if not page["results"]:
                break
            time.sleep(0.12)
        return
    n = 1
    while True:
        page = get(path, {**params, "per-page": 100, "page": n})
        if n == 1 and page["meta"].get("count", 0) > 10000:
            raise RuntimeError(
                f"query has {page['meta']['count']} results (>10,000): set OPENALEX_API_KEY "
                "(free key at https://openalex.org/settings/api)")
        if progress:
            progress(len(page["results"]), page["meta"].get("count"))
        yield from page["results"]
        if len(page["results"]) < 100:
            break
        n += 1
        time.sleep(0.12)


def citing_of(batch, sel, progress=None):
    """Works citing any target in batch; without an API key, batches whose result
    count exceeds 10,000 are split in halves (page paging limit)."""
    flt = "cites:" + "|".join(batch)
    if not os.environ.get("OPENALEX_API_KEY") and len(batch) > 1:
        cnt = get("/works", {"filter": flt, "per-page": 1, "select": "id"})["meta"]["count"]
        if cnt > 10000:
            h = len(batch) // 2
            yield from citing_of(batch[:h], sel, progress)
            yield from citing_of(batch[h:], sel, progress)
            return
    yield from paged("/works", {"filter": flt, "select": sel}, progress)


def plos_one_source_id():
    res = get("/sources", {"filter": f"issn:{PLOS_ONE_ISSN}"})["results"]
    if not res:
        raise RuntimeError("PLOS ONE not found in OpenAlex sources")
    return short(res[0]["id"])


def author_ids(w):
    return sorted({short(a["author"]["id"]) for a in w.get("authorships") or []
                   if a.get("author") and a["author"].get("id")})


def fetch_targets(out, years):
    path = os.path.join(out, "targets.jsonl.gz")
    if os.path.exists(path):
        print("targets: already downloaded")
        return path
    sid = plos_one_source_id()
    print("PLOS ONE source:", sid)
    tmp = path + ".part"
    n = 0
    with gzip.open(tmp, "wt") as fh:
        for y in years:
            flt = f"primary_location.source.id:{sid},publication_year:{y},type:article"
            got = [0]

            def prog(k, total, y=y):
                got[0] += k
                status(f"stage 1/2, PLOS ONE {y}: {got[0]}/{total} articles", force=True)

            for w in paged("/works", {"filter": flt,
                                      "select": "id,doi,publication_date,authorships,cited_by_count,primary_topic"},
                           prog):
                topic = w.get("primary_topic") or {}
                rec = dict(id=short(w["id"]), doi=w.get("doi"), date=w.get("publication_date"),
                           authors=author_ids(w), cited_by=w.get("cited_by_count"),
                           field=short((topic.get("field") or {}).get("id")))
                fh.write(json.dumps(rec) + "\n")
                n += 1
            print(f"\n  year {y}: cumulative {n} targets", flush=True)
    os.replace(tmp, path)
    return path


def fetch_citing(out, batch_size=50):
    targets = [json.loads(l)["id"] for l in gzip.open(os.path.join(out, "targets.jsonl.gz"), "rt")]
    tset = set(targets)
    batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
    bdir = os.path.join(out, "citing")
    os.makedirs(bdir, exist_ok=True)
    # each finished batch is one file, written atomically; reruns skip them
    seen = set()
    done = set()
    for fn in sorted(os.listdir(bdir)):
        if fn.endswith(".jsonl.gz"):
            done.add(int(fn.split("_")[1].split(".")[0]))
            with gzip.open(os.path.join(bdir, fn), "rt") as fh:
                seen.update(json.loads(l)["id"] for l in fh)
    sel = ("id,doi,publication_date,type,referenced_works,referenced_works_count,"
           "authorships,primary_location,primary_topic,cited_by_count")
    t0 = time.time()
    todo = len(batches) - len(done)
    print(f"stage 2/2: {len(batches)} batches of {batch_size} articles, {len(done)} already done, "
          f"{todo} to go", flush=True)
    nrun = 0
    for b, batch in enumerate(batches):
        if b in done:
            continue
        nnew = 0
        got = [0]

        def prog(k, total, b=b):
            got[0] += k
            el = time.time() - t0
            eta = el / nrun * (todo - nrun) if nrun else float("nan")
            status(f"stage 2/2: batch {b + 1}/{len(batches)} [{got[0]}/{total} records] | "
                   f"{len(done)}/{len(batches)} done, {len(seen)} citing works | "
                   f"elapsed {fmt_time(el)}" + (f", ~{fmt_time(eta)} left" if nrun else ""), force=True)
        part = os.path.join(bdir, f"batch_{b:05d}.part")
        with gzip.open(part, "wt") as fh:
            for w in citing_of(batch, sel, prog):
                wid = short(w["id"])
                if wid in seen:
                    continue
                seen.add(wid)
                refs = [short(r) for r in (w.get("referenced_works") or [])]
                loc = w.get("primary_location") or {}
                src = (loc.get("source") or {})
                topic = w.get("primary_topic") or {}
                rec = dict(id=wid, doi=w.get("doi"), date=w.get("publication_date"),
                           type=w.get("type"), k=w.get("referenced_works_count"),
                           k_list=len(refs), cites=[r for r in refs if r in tset],
                           authors=author_ids(w), source=short(src.get("id")),
                           source_type=src.get("type"),
                           field=short((topic.get("field") or {}).get("id")),
                           cited_by=w.get("cited_by_count"))
                fh.write(json.dumps(rec) + "\n")
                nnew += 1
        os.replace(part, os.path.join(bdir, f"batch_{b:05d}.jsonl.gz"))
        done.add(b)
        nrun += 1
        el = time.time() - t0
        eta = el / nrun * (todo - nrun)
        status(f"stage 2/2: {len(done)}/{len(batches)} batches done ({100 * len(done) / len(batches):.0f}%), "
               f"{len(seen)} citing works | elapsed {fmt_time(el)}, ~{fmt_time(eta)} left", force=True)
        # keep one permanent line every 10 batches as a log
        if nrun % 10 == 0 or len(done) == len(batches):
            print(flush=True)
    return bdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--years", default="2010", help="e.g. 2010 or 2010-2012")
    a = ap.parse_args()
    if "-" in a.years:
        y0, y1 = map(int, a.years.split("-"))
        years = list(range(y0, y1 + 1))
    else:
        years = [int(a.years)]
    os.makedirs(a.out, exist_ok=True)
    fetch_targets(a.out, years)
    fetch_citing(a.out)
    print("\ndone: data are in", os.path.abspath(a.out))


if __name__ == "__main__":
    sys.exit(main())
