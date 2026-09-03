# TAX-DPD final-blind-v2 artifacts

This directory is published on the `artifacts/final-blind-v2` branch of
`What410802/TAX-DPD`. GitHub's public-fork LFS endpoint rejects new objects
for this repository, so these binaries are stored in ordinary Git. The largest
file is below GitHub's per-file limit.

## Contents

- `2x2/`: PointNet++/Utonia × DDPM/FM checkpoints, training histories,
  target-free predictions, evaluator reports and the four-cell summary.
- `utonia-feature-cache/`: v2 frozen Utonia feature cache and its manifest,
  generated for the exact ordered export used by the four-cell experiment.
- `wandb-retrain-20260903/`: four fresh ten-epoch retrains with TensorBoard
  events, target-free test predictions and fixed-index evaluator reports.

All four runs use the v2 72/12/12 grouped split, `seed=20260903`, ten epochs,
candidate-0 evaluation and the prediction-only contract. The reports are a
12-sample pilot and retain `quality_claim=false`.

## Boundary

The cache is not a general Utonia checkpoint. Running the model still requires
the code commit `research/placegen-2x2` at `b301713`, the locked Pixi
environment, the externally supplied Utonia pretrained weights and the B300
CUDA extension compatibility fix. Those dependencies are intentionally not
embedded in this artifact branch.

After cloning, verify checkpoint/evaluation hashes in the PlaceGen reproduction
guide. If GitHub later enables LFS for this fork, the branch can be migrated
without changing the artifact paths.

## W&B retrain runs

| run directory | W&B run |
| --- | --- |
| `pn2-fm/` | https://wandb.ai/models-xi-an-jiaotong-university-9458/PlaceGen/runs/ci9jgg4d |
| `pn2-ddpm/` | https://wandb.ai/models-xi-an-jiaotong-university-9458/PlaceGen/runs/twrr3nrz |
| `utonia-fm/` | https://wandb.ai/models-xi-an-jiaotong-university-9458/PlaceGen/runs/d3z79yz0 |
| `utonia-ddpm/` | https://wandb.ai/models-xi-an-jiaotong-university-9458/PlaceGen/runs/avtll0fy |

The retrain code is `What410802/TAX-DPD` commit `6e9feb2` (the logger fix and
TensorBoard lock). Each run directory contains `manifest.json`, checkpoint,
prediction/evaluation JSON, `run.log` and a TensorBoard event file. The
evaluation reports use the same 12 held-out `final-blind-v2` samples and keep
`planner_called=false` and `simulator_called=false`.
