# Paper analyses

Every number in the paper comes from one of these scripts. They read aggregate inputs from `results/` and write to
`results/paper_outputs/` (override with `ECHO_RESULTS` / `ECHO_OUT`). Run from this directory.

| Script | Needs | Produces | Paper |
|---|---|---|---|
| `converged_probe_gain.py` | adapted Gram matrices + frozen embeddings (GPU pipeline outputs, not distributed) | `results/probes/converged/*.csv` | frozen / adapted accuracy, gain |
| `adaptation_geometry.py` | same | `results/geometry/adaptation/*.csv` | shared offset, change in uniformity |
| `finding2_frozen_probes.py` | frozen embeddings | `results/probes/frozen_knn_linear.csv`, `frozen_linear_iterations.csv` | Finding 2 probes |
| `finding1.py` | `results/geometry/rsa_frozen/` (+ `ECHO_VESSEL_RSA` for the vessel matrix) | cross-domain stability, taxonomy Mantel tests, leave-one-out | Finding 1 |
| `finding2.py` | `results/probes/`, `results/geometry/` | sign test, Bird-MAE/AudioMAE ratios, uniformity correlations | Finding 2 |
| `finding3_gain_regression.py` | `results/probes/converged/` | regression, bootstrap CIs, within-domain ρ, sensitivity variants | Finding 3 |
| `finding3_predictor_search.py` | same + frozen probes, geometry, parameter counts | exhaustive 1–3-predictor search with permutation null | Finding 3 |
| `finding3_transferability.py` | `results/probes/transferability_estimators.csv` | LogME and kNN-reference vs gain | Finding 3 |
| `finding3_reorganization.py` | `results/geometry/` | shared offset, ΔRSA, preserved ordering, self-RSA links | Finding 3 |
| `make_appendix.py` | all of the above | `APPENDIX.md`, `figures/appendix/` | appendix |

Everything after the first three rows runs on a laptop from the files in `results/`:

```
cd scripts/paper
python finding1.py && python finding2.py && python finding3_gain_regression.py && python finding3_predictor_search.py \
  && python finding3_transferability.py && python finding3_reorganization.py && python make_appendix.py
```

Runtimes on 16 CPU cores: predictor search about 1 h (2,000 permutations of the full search), Finding 1 about 30 min,
reorganization about 7 min, the rest under 2 min.

The vessel domain is analysed with the same code on a restricted server. Its inputs are not distributed; the aggregate
statistics it contributes are in `results/finding1/` and `additional_results/`.
