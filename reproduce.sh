#!/usr/bin/env bash
# Full pipeline of the paper, from download to figures.
# Requires an OpenAlex API key:  export OPENALEX_API_KEY="..."
# Downloads are resumable: if a step stops (e.g. HTTP 429 when the daily quota is used up),
# rerun it later with the same arguments.
set -euo pipefail
: "${OPENALEX_API_KEY:?set OPENALEX_API_KEY first}"

# 1. Cohort and citing works (~3-4 h)
python src/fetch_openalex.py --out data --years 2010

# 2. Co-citation sample, stratified by reference-list length (~17.5k requests)
python src/fetch_cocite.py --data data --per-bin 2500

# 3. Placebos (drawn from data/cocite.csv: run after step 2 is final)
python src/fetch_cocite_placebo.py --data data --per-bin 1000 --match neighbor
python src/fetch_cocite_placebo.py --data data --per-bin 1000 --match field

# 4. Estimates
python src/cocite_analyze.py --data data --out results_cocite | tee results_cocite.log
python src/cocite_analyze.py --data data --placebo | tee -a results_cocite.log
python src/cocite_analyze.py --data data --placebo cocite_placebo.csv | tee -a results_cocite.log

# 5. Figures
python figures/make_fig1.py
python figures/make_fig2.py results_cocite/cocite_numbers.json
