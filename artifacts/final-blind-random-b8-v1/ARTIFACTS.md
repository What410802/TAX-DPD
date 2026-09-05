# B=8 random-sampler exploratory artifacts

This artifact block records the update-aligned batch-size rerun requested for the
PlaceGen/TAX3Dv2 pilot. It is not a new blind result: the
`rack-plate-final-blind-v2-random` test split had already been opened by the
B=1 audit.

## Provenance

- PlaceGen random-sampler artifact source: `What410802/placegen`, branch
  `research/final-blind-random-v1`, artifact commit `7637c00`.
- TAX-DPD trainer: `What410802/TAX-DPD`, branch `research/placegen-2x2`, commit
  `eb8d9d7`.
- Data manifest SHA-256:
  `e3578a2459271f8c25249171f236aaf6debbb78b43bdc06f7ab8087d32defb73`.
- Inference manifest SHA-256:
  `6821638818dea35cfb0bcb6074205bbc99f65d47b097760abd11be37abcdb248`.
- Utonia feature-cache manifest SHA-256:
  `b1e78163d1f7d60a856d23f088601cf042d246e61e30bc4f4a0e66cf23846893`.

The four runs use FP32, `batch_size=8`, `validation_batch_size=8`, 80 epochs,
72/12 train/validation samples, `lr=1e-4`, and seed `20260905`. They perform
9 optimizer updates per epoch, 720 updates total, matching the 10-epoch B=1
pilot's update count. Checkpoint selection uses validation only; prediction is
target-free and stops before planner/simulator execution.

## Results

| model | checkpoint SHA-256 | best validation loss | test translation mean (mm) | rotation mean (deg) | ordered RMSE mean (mm) | symmetric Chamfer mean (mm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PointNet++ + FM | `82b9f9705c70c46fae592b41e87b96f6670a984040bf1c9f613e9838c8457b31` | 0.0002676 | 16.644 | 0.081 | 16.645 | 9.655 |
| Utonia cached + FM | `b92e17183626b26f13e98e78e75b28474681afed043dfb113ee3e2560961c29f` | 0.0002249 | 12.249 | 0.105 | 12.252 | 8.337 |
| PointNet++ + DDPM | `c1d9198969c12965042d9b5ede95d51edb60217d309024b09a349953948df454` | 0.0562436 | 33.428 | 2.144 | 33.528 | 21.369 |
| Utonia cached + DDPM | `8a488f08a4fde3186ee2eb54d55434e76a8615c396c9f966601b421386641369` | 0.0194040 | 17.783 | 2.068 | 18.041 | 10.374 |

For each row, the sibling `*-test-predictions.json` and
`*-test-evaluation.json` retain complete per-sample evidence. All reports set
`planner_called=false`, `simulator_called=false`, and `quality_claim=false`.
The evaluator still compares against one canonical target pose per sample; it
does not implement the task's full “any valid rack slot” success set.

## Interpretation

The 10-epoch B=8 run has only 90 updates and substantially undertrains DDPM.
After matching the B=1 pilot's 720 updates, DDPM improves to approximately
33.43 mm (PointNet++) and 17.78 mm (Utonia), while FM remains in the same
15–16 mm range for PointNet++ and improves to 12.25 mm with Utonia. This is
evidence to report batch size together with total optimizer updates; it is not
evidence that B=8 is universally better or worse.

## Reproduction boundary

The files are ordinary Git blobs (largest file below GitHub's per-file limit).
They depend on the TAX-DPD code/environment, the B300-compatible PyTorch3D
and PyG extensions, and the frozen Utonia feature cache contract. No planner,
Isaac, cuRobo, MuJoCo, or robot motion is included in this artifact block.
