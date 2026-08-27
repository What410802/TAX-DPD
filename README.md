# Disentangled Point Diffusion for Precise Object Placement

📌 Official code release for our ICRA 2026 paper:

**Disentangled Point Diffusion for Precise Object Placement**\
*Lyuxing He, Eric Cai, Shobhit Aggarwal, Jianjun Wang, David Held*\
[Project Page](https://3dgp-icra2026.github.io/) | [arXiv](https://arxiv.org/abs/2604.11793)

## 📚 Citation

```bibtex
@article{he2026disentangled,
  title={Disentangled Point Diffusion for Precise Object Placement},
  author={He, Lyuxing and Cai, Eric and Aggarwal, Shobhit and Wang, Jianjun and Held, David},
  journal={arXiv preprint arXiv:2604.11793},
  year={2026}
}
```

## ✅ Roadmap

- [x] Initial code release
- [ ] Installation instructions
- [ ] Training instructions
- [ ] Evaluation instructions
- [ ] Pre-trained checkpoints

## PlaceGen held-out rack/plate path

Branch `research/placegen-rack-generalization` adds a deliberately narrow
integration for PlaceGen's fixed-asset, fixed-slot, held-out-pose dataset.  The
implemented capability is always named
`TAX-DPD-reconstructed-fixed-frame-w/o-GMM`: `tax3d_v2.py` is reconstructed,
and this path does not reproduce the unpublished official TAX-DPD module or its
GMM stage.

The trust boundaries are separate:

- training accepts only PlaceGen profile
  `placegen.taxdpd-rpdiff-grouped/0.1`, validates 72/12/12 physical setup
  groups by default, and binds every checkpoint to the dataset-manifest and
  split-assignment SHA-256 values;
- checkpoint selection uses only validation candidate 0 ordered point RMSE;
  the test loader is never constructed during training;
- held-out prediction accepts only the target-free
  `inference/manifest.json` profile.  Passing the supervision-bearing top-level
  manifest fails closed;
- online PlaceGen calls use a second strict request/response seam containing
  exactly six observation arrays and no target pose or goal point cloud.

Typical commands are:

```bash
pixi run --locked placegen-train-grouped -- \
  --manifest /ARTIFACT/taxdpd/manifest.json \
  --checkpoint-dir /ARTIFACT/run/checkpoints \
  --report /ARTIFACT/run/training-report.json \
  --epochs 4 --device cuda

pixi run --locked placegen-predict-grouped -- \
  --inference-manifest /ARTIFACT/taxdpd/inference/manifest.json \
  --checkpoint /ARTIFACT/run/checkpoints/epoch-XXXX.pt \
  --output-json /ARTIFACT/run/test-predictions.json \
  --candidates 1 --device cuda

pixi run --locked placegen-predict-request -- \
  --request /ARTIFACT/request/request.json \
  --checkpoint /ARTIFACT/run/checkpoints/epoch-XXXX.pt \
  --output /ARTIFACT/request/response.json \
  --device cuda
```

Use `best_checkpoint` from `training-report.json` for prediction; do not pick a
checkpoint using test results.  Training and inference reports include losses,
gradient norms, wall time, peak allocated VRAM, reload checks, and provenance
hashes.  They keep `quality_claim=false` until the separately isolated held-out
evaluation has been completed.
