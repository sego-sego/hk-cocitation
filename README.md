# Square-root dilution of value transfer through citations — data and code

Code and data for

> A. N. Pisarchik and A. E. Hramov, *Square-root dilution of value transfer through citations*
> (submitted to Chaos, Solitons & Fractals, 2026).

The paper measures the transfer function $h(k)$ of a broadcast event — the share of value that
each of $k$ recipients receives — for citations. A citing work $j$ with $k$ references is a
broadcast; the transfer to a reference $i$ is measured by the number $Y_{ij}$ of later works that
cite both $j$ and $i$. With fixed effects of the cited article, $Y_{ij}\propto k_j^{-\gamma}$ with
$\gamma = 0.51 \pm 0.04$ for PLOS ONE 2010 articles.

## Contents

| path | what it is |
|---|---|
| `src/fetch_openalex.py` | Stage 1 download: PLOS ONE 2010 articles and every work citing them (OpenAlex API) |
| `src/fetch_cocite.py` | Stage 2 download: co-citation counts $Y_{ij}$ for a $k$-stratified sample of pairs |
| `src/fetch_cocite_placebo.py` | Placebo download: co-citation of $j$ with an article it does not cite (topical neighbor or same field) |
| `src/cpois.py` | Poisson regression with article fixed effects (conditional likelihood) and clustered sandwich errors |
| `src/cocite_analyze.py` | All estimates of the paper: $\gamma$, bins, robustness, power-vs-club test, placebo, within-pair difference |
| `figures/make_fig1.py` | Fig. 1 (schematic and regimes) |
| `figures/make_fig2.py` | Fig. 2 (bin estimates with power-law and club fits) |
| `validation/` | Synthetic data with known $h(k)$ and an end-to-end test of the pipeline with a mock API (Appendix B) |
| `appendix_dynamic/` | The dynamic citation-growth design and its indirect-inference calibration (Appendix A) |
| `data/README.md` | Description of the data files deposited on Zenodo |
| `reproduce.sh` | The full sequence of commands |
| `make_release.sh` | Packs the downloaded data and results into the Zenodo deposit with checksums |

## Requirements

Python ≥ 3.10 with `numpy`, `pandas`, `scipy`, `matplotlib` (versions used: see
`requirements.txt`). No econometrics packages are needed. The downloads use only the Python
standard library.

An OpenAlex API key (free, https://openalex.org/settings/api) is required for cursor paging:

```bash
export OPENALEX_API_KEY="..."      # never commit the key
```

## Reproducing the results

From the deposited data (no API access needed):

```bash
# unpack the Zenodo archive into ./data, then
python src/cocite_analyze.py --data data --out results_cocite      # main results, Tables I-II, shape test
python src/cocite_analyze.py --data data --placebo                 # Table III (topical-neighbor placebo)
python src/cocite_analyze.py --data data --placebo cocite_placebo.csv   # same-field placebo
python figures/make_fig1.py
python figures/make_fig2.py results_cocite/cocite_numbers.json
```

From scratch (new download): see `reproduce.sh`. OpenAlex is updated continuously, so a new
download gives slightly different counts. The sample of pairs is drawn with fixed seeds
(`--seed` in the scripts), but it is drawn from the current index.

Validation on synthetic data (Appendix B):

```bash
python validation/validate_cocite.py --gamma 0.0
python validation/validate_cocite.py --gamma 0.6
```

Dynamic design (Appendix A; slow, ~1 h):

```bash
PYTHONPATH=src python appendix_dynamic/analyze.py --data data --out results_dynamic --quick
```

## Notes on the download

* The co-citation queries (`cites:j,cites:i`) are slow on the OpenAlex side (~5 s each) and count
  against the daily quota of the API key. The scripts are resumable: rerun the same command with
  the same `--per-bin` and they skip pairs already retrieved.
* Queries are issued in the order of the stratified draw, so an interrupted download leaves the
  rarest bins (short reference lists) incomplete; within each bin the retrieved pairs remain a
  random draw.
* Run `fetch_cocite_placebo.py` before extending `cocite.csv`: the placebo sample is drawn from the
  pairs in `cocite.csv`.

## License

Code: MIT (see `LICENSE`). Data: derived from OpenAlex, which is released under CC0; the deposited
data are released under CC0 as well.

## Citation

See `CITATION.cff`.
