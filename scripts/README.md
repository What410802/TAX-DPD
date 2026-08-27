# Scripts

If you write some scripts which are meant to be run stand-alone, and not imported as part of the library, put them in this directory.

## PlaceGen rack/plate generalization

`placegen_train_grouped.py` is the supervision-bearing entry point.  It
validates PlaceGen's grouped manifest and every declared training NPZ before it
constructs only the train and validation `RPDiffDataset` instances.  It runs at
least two epochs and two optimizer steps, writes one immutable checkpoint per
epoch, selects the best epoch only by validation candidate-0 ordered RMSE, then
reloads the selected model and optimizer state before reporting success.

`placegen_predict_grouped.py` is the offline held-out entry point.  It accepts
only `.../inference/manifest.json`, verifies its split-assignment and native
dataset identities against the checkpoint, and predicts every test sample
without opening the top-level training manifest.

`placegen_predict_request.py` is the online PlaceGen process seam.  Invoke it as
follows (the final `--` makes later arguments belong to the task):

```bash
pixi run --manifest-path /ABS/TAX-DPD/pixi.toml --locked placegen-predict-request -- \
  --request /ABS/request.json \
  --output /ABS/response.json \
  --checkpoint /ABS/checkpoint.pt
```

The request fields are exactly:

```text
profile, request_id, model, checkpoint_sha256,
input_npz {path, sha256}, seed, candidate_count
```

The NPZ fields are exactly:

```text
action_indices:int64
anchor_indices:int64
child_points_world:float32
parent_points_world:float32
scene_center_world:float64
source_world_from_object:float64
```

The response echoes the request/checkpoint/input identities and returns ordered
world-frame poses.  `score_kind=negative_ordered_rigid_fit_rmse_m` measures only
how rigidly a sampled goal cloud fits the observed plate; it is not a calibrated
placement-success probability.
