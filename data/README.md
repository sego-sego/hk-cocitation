# Data files

All data were retrieved from the OpenAlex API (https://openalex.org) on 25–26 October 2026 and are
released under CC0, as is OpenAlex. Identifiers are OpenAlex work, author and source IDs without the
`https://openalex.org/` prefix (e.g. `W2087591041`).

## `targets.jsonl.gz` — the cohort (6 691 lines)

PLOS ONE (OpenAlex source `S202381698`), publication year 2010, type `article`. One JSON object per line:

| field | type | meaning |
|---|---|---|
| `id` | string | OpenAlex work ID |
| `doi` | string | DOI URL |
| `date` | string | publication date (YYYY-MM-DD) |
| `authors` | list of strings | OpenAlex author IDs |
| `cited_by` | int | citation count at retrieval |
| `field` | string | OpenAlex field ID of the primary topic |

## `citing/batch_NNNNN.jsonl.gz` — works citing the cohort (505 878 works)

Every work that cites at least one target, in batches of 50 targets. A work citing targets in
several batches appears once, in the first batch that retrieved it.

| field | type | meaning |
|---|---|---|
| `id`, `doi`, `date`, `type` | | OpenAlex ID, DOI, publication date, document type |
| `k` | int | `referenced_works_count` (references matched by OpenAlex) |
| `k_list` | int | length of the returned `referenced_works` list |
| `cites` | list of strings | the targets this work cites |
| `authors` | list of strings | OpenAlex author IDs (used to flag self-citations) |
| `source`, `source_type` | string | primary source ID and type |
| `field` | string | OpenAlex field ID of the primary topic |
| `cited_by` | int | citation count of this work at retrieval ($n_j$) |

## `cocite.csv` — co-citation sample (13 544 pairs; Sec. III of the paper)

Non-self pairs $(i, j)$ with $j$ published up to 2018, stratified by $k_j$ (bins 1–15, 16–25, 26–35,
36–50, 51–75, 76–120, >120; up to 2 500 pairs per bin, seed 20261025).

| column | meaning |
|---|---|
| `i` | target (cited article) |
| `j` | citing work |
| `k` | reference-list length of `j` |
| `date`, `type`, `field`, `source` | of `j` |
| `n_j` | citation count of `j` |
| `y` | number of works citing both `j` and `i` ($Y_{ij}$), from the filter `cites:j,cites:i` |

## `cocite_placebo_nb.csv` — topical-neighbor placebo (Sec. V D, Table III)

Pairs from `cocite.csv` (up to 1 000 per bin, seed 20261026). Columns as in `cocite.csv`, plus

| column | meaning |
|---|---|
| `i_placebo` | the PLOS ONE 2010 article most often co-cited with `i` by other works and not cited by `j` |
| `y_placebo` | number of works citing both `j` and `i_placebo` |

## `cocite_placebo.csv` — same-field placebo (Sec. V D)

As `cocite_placebo_nb.csv`, with `i_placebo` a random PLOS ONE 2010 article of the same OpenAlex
field as `i`, not cited by `j`.

## `results_cocite/cocite_numbers.json`

All estimates reported in the paper, as written by `src/cocite_analyze.py`.

## Checksums

`SHA256SUMS` lists SHA-256 checksums of all files in the deposit.
