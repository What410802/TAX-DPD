"""
TAX3Dv2 / TAX-DPD model implementation.

Reconstructed from:
  - configs/model/tax3dv2.yaml        (architecture params)
  - src/non_rigid/models/dit/models.py (TAX3Dv2_FixedFrame_Token_DiT,
                                        TAX3Dv2_MuFrame_DiT)
  - src/non_rigid/models/dit/diffusion/ (ddrd_separate, mu diffusion)
  - src/non_rigid/models/df_base.py   (training loop patterns)
  - src/non_rigid/utils/script_utils.py (expected exports)

Exports required by script_utils.py:
  - TAX3Dv2Network
  - TAX3Dv2FixedFrameModule
  - TAX3Dv2MuFrameModule
"""

from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import omegaconf
import torch
import wandb
from diffusers import get_cosine_schedule_with_warmup
from torch import nn, optim

from non_rigid.metrics.error_metrics import get_pred_pcd_rigid_errors
from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse
from non_rigid.models.dit.diffusion import (
    create_diffusion_ddrd_separate,
    create_diffusion_mu,
)
from non_rigid.models.dit.models import (
    TAX3Dv2_FixedFrame_Token_DiT,
    TAX3Dv2_MuFrame_DiT,
)
from non_rigid.models.flow_matching import (
    flow_matching_loss,
    integrate_velocity,
)
from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.utils.pointcloud_utils import expand_pcd


# ─── helpers ──────────────────────────────────────────────────────────────────

def _hidden_size(model_cfg: omegaconf.DictConfig) -> int:
    """xS DiT hidden size: 132 with rel_pos (divisible by 3), 128 otherwise."""
    return 132 if model_cfg.rel_pos else 128


# ─── network ──────────────────────────────────────────────────────────────────

class TAX3Dv2Network(nn.Module):
    """
    Thin nn.Module wrapper that instantiates the right TAX3Dv2 DiT backbone
    based on model_cfg.frame_type.

    frame_type == "fixed"  →  TAX3Dv2_FixedFrame_Token_DiT
    frame_type == "mu"     →  TAX3Dv2_MuFrame_DiT
    """

    def __init__(self, model_cfg: omegaconf.DictConfig):
        super().__init__()
        self.model_cfg = model_cfg
        hidden_size = _hidden_size(model_cfg)

        common_kwargs = dict(
            in_channels=model_cfg.in_channels,
            hidden_size=hidden_size,
            depth=5,       # "xS" architecture (DiT_PointCloud_xS uses depth=5)
            num_heads=4,
            learn_sigma=model_cfg.learn_sigma,
            model_cfg=model_cfg,
        )

        if model_cfg.frame_type == "fixed":
            self.dit = TAX3Dv2_FixedFrame_Token_DiT(**common_kwargs)
        elif model_cfg.frame_type == "mu":
            self.dit = TAX3Dv2_MuFrame_DiT(**common_kwargs)
        else:
            raise ValueError(
                f"Unknown frame_type '{model_cfg.frame_type}'. "
                "Expected 'fixed' or 'mu'."
            )

    def forward(self, xr_t: torch.Tensor, xs_t: torch.Tensor,
                t: torch.Tensor, **kwargs):
        """
        Args:
            xr_t: noised reference-frame centroid  [B, 3, 1]
            xs_t: noised shape (residual)          [B, 3, N]
            t:    diffusion timestep               [B]
            **kwargs: forwarded to the DiT (y, x0, rel_pos, finetune_frame …)
        Returns:
            (xr_out, xs_out) – predicted noise/x0 for frame and shape
        """
        return self.dit(xr_t, xs_t, t, **kwargs)


# ─── base training module ─────────────────────────────────────────────────────

class _TAX3Dv2BaseModule(L.LightningModule):
    """Shared optimiser, logging and WTA scaffolding for both frame variants."""

    def __init__(self, network: TAX3Dv2Network, cfg: omegaconf.DictConfig):
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.training_cfg = cfg.training

        self.lr = cfg.training.lr
        self.weight_decay = cfg.training.weight_decay
        self.num_training_steps = cfg.training.num_training_steps
        self.lr_warmup_steps = cfg.training.lr_warmup_steps
        self.batch_size = cfg.training.batch_size
        self.val_batch_size = cfg.training.val_batch_size
        self.sample_size = cfg.training.sample_size
        self.sample_size_anchor = cfg.training.sample_size_anchor
        self.diff_train_steps = cfg.model.diff_train_steps
        self.diff_inference_steps = cfg.model.diff_inference_steps
        self.num_wta_trials = cfg.training.num_wta_trials
        self.noise_scale = cfg.model.diff_noise_scale

    # ── optimiser ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    # ── helpers (subclasses override _get_x_start) ───────────────────────────

    def _model_kwargs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Build DiT conditioning kwargs from a batch."""
        result = {
            "y":  batch["pc_anchor"].to(self.device).permute(0, 2, 1),
            "x0": batch["pc_action"].to(self.device).permute(0, 2, 1),
        }
        if str(getattr(self.model_cfg, "point_encoder", "")) == "utonia_cached":
            sample_id = batch.get("sample_id")
            if sample_id is not None:
                if isinstance(sample_id, str):
                    result["_utonia_sample_ids"] = [sample_id]
                else:
                    result["_utonia_sample_ids"] = [str(value) for value in sample_id]
        return result

    def _get_x_start(self, batch: Dict) -> torch.Tensor:
        """Return target point cloud [B, 3, N] for the diffusion loss."""
        raise NotImplementedError

    # ── training ─────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        self.train()
        x_start = self._get_x_start(batch)            # [B, 3, N]
        t = torch.randint(
            0, self.diff_train_steps, (x_start.shape[0],), device=self.device
        ).long()
        model_kwargs = self._model_kwargs(batch)

        loss_dict = self.diffusion.training_losses(
            self.network, x_start, t, model_kwargs
        )
        loss = loss_dict["loss"].mean()

        self.log("train/loss", loss, prog_bar=True, add_dataloader_idx=False)

        do_extra = (
            self.global_step > 0
            and self.global_step % self.training_cfg.additional_train_logging_period == 0
        )
        if do_extra:
            with torch.no_grad():
                wta = self._predict_wta(batch, num_trials=self.num_wta_trials)
            self.log_dict(
                {
                    "train_wta/rmse":    wta["rmse_wta"].mean(),
                    "train_wta/cos_sim": wta["cos_sim_wta"].mean(),
                    "train/rmse":        wta["rmse"].mean(),
                    "train/cos_sim":     wta["cos_sim"].mean(),
                },
                add_dataloader_idx=False,
            )
            # RPDiff does not provide distractor poses, so its optional rigid
            # error metric cannot be computed with the default distractor_min.
            try:
                errors = get_pred_pcd_rigid_errors(
                    batch=batch,
                    pred_xyz=wta["pred_actions_wta"],
                    error_type=self.training_cfg.prediction_error_type,
                )
                self.log_dict(
                    {
                        "train/error_t_mean": errors["error_t_mean"],
                        "train/error_R_mean": errors["error_R_mean"],
                    },
                    add_dataloader_idx=False,
                )
            except (KeyError, AssertionError):
                pass
            # Visualisation
            viz_idx = np.random.randint(0, batch["pc"].shape[0])
            wandb.log({
                "train_wta/predicted_vs_gt": viz_predicted_vs_gt(
                    pc_pos_viz=batch["pc"][viz_idx, :, :3],
                    pc_action_viz=batch["pc_action"][viz_idx, :, :3],
                    pc_anchor_viz=batch["pc_anchor"][viz_idx, :, :3],
                    pred_action_viz=wta["pred_actions_wta"][viz_idx, :, :3],
                )
            })
        return loss

    # ── validation ───────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        with torch.no_grad():
            wta = self._predict_wta(batch)
        self.log_dict(
            {
                # Key names must match the ModelCheckpoint monitors in
                # scripts/train.py: "val_rmse_wta_0" and "val_rmse_0".
                # (df_base.py uses val_wta_rmse_* but it is a legacy file that
                # script_utils.py never imports -- tax3d.py is the live one.)
                f"val_rmse_wta_{dataloader_idx}":    wta["rmse_wta"].mean(),
                f"val_cos_sim_wta_{dataloader_idx}": wta["cos_sim_wta"].mean(),
                f"val_rmse_{dataloader_idx}":        wta["rmse"].mean(),
                f"val_cos_sim_{dataloader_idx}":     wta["cos_sim"].mean(),
            },
            prog_bar=True,
            add_dataloader_idx=False,
        )
        # Rigid placement errors (optional — skipped if the batch lacks required
        # keys, e.g. RPDiff has no distractor list for distractor_min).
        try:
            errors = get_pred_pcd_rigid_errors(
                batch=batch,
                pred_xyz=wta["pred_actions_wta"],
                error_type=self.training_cfg.prediction_error_type,
            )
            self.log_dict(
                {
                    f"val/error_t_mean_{dataloader_idx}": errors["error_t_mean"],
                    f"val/error_R_mean_{dataloader_idx}": errors["error_R_mean"],
                },
                add_dataloader_idx=False,
            )
        except (KeyError, AssertionError):
            pass
        # Visualisation
        viz_idx = np.random.randint(0, batch["pc"].shape[0])
        wandb.log({
            f"val_wta/predicted_vs_gt_{dataloader_idx}": viz_predicted_vs_gt(
                pc_pos_viz=batch["pc"][viz_idx, :, :3],
                pc_action_viz=batch["pc_action"][viz_idx, :, :3],
                pc_anchor_viz=batch["pc_anchor"][viz_idx, :, :3],
                pred_action_viz=wta["pred_actions_wta"][viz_idx, :, :3],
            )
        })
        return {"loss": wta["rmse_wta"]}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        wta = self._predict_wta(batch)
        return {
            "cos_sim_wta": wta["cos_sim_wta"],
            "cos_sim":     wta["cos_sim"],
            "rmse_wta":    wta["rmse_wta"],
            "rmse":        wta["rmse"],
        }

    def _predict_wta(self, batch: Dict, num_trials: Optional[int] = None) -> Dict:
        raise NotImplementedError


# ─── Fixed-frame module ───────────────────────────────────────────────────────

class TAX3Dv2FixedFrameModule(_TAX3Dv2BaseModule):
    """
    TAX3Dv2 training module for the "fixed" prediction frame.

    The reference frame is the centroid of the *goal* point cloud.
    The diffusion decomposes x_start internally:
        xr_start = mean(x_start)  →  frame  (centroid)
        xs_start = x_start - xr  →  shape  (zero-mean residual)
    Both are diffused independently with optional rotation noise on xs.
    """

    def __init__(self, network: TAX3Dv2Network, cfg: omegaconf.DictConfig):
        super().__init__(network, cfg)
        self.diffusion = create_diffusion_ddrd_separate(
            timestep_respacing=None,
            noise_schedule=cfg.model.diff_noise_schedule,
            learn_sigma=cfg.model.learn_sigma,
            diffusion_steps=cfg.model.diff_train_steps,
            # Passed through verbatim: _time_based_weights() dispatches on the
            # string ("even" | "linear" | "sigmoid") and raises on anything else.
            time_based_weighting=cfg.model.time_based_weighting,
            rotation_noise_scale=cfg.model.diff_rotation_noise_scale,
            zero_shape=cfg.model.zero_shape,
        )

    def _get_x_start(self, batch: Dict) -> torch.Tensor:
        # goal point cloud in scene frame, channel-first [B, 3, N]
        return batch["pc"].to(self.device).permute(0, 2, 1)

    def _model_kwargs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return self._ddpm_model_kwargs(super()._model_kwargs(batch))

    def _ddpm_model_kwargs(
        self,
        model_kwargs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Consume Utonia cache identity before calling the DiT network."""

        network_kwargs = dict(model_kwargs)
        sample_ids = network_kwargs.pop("_utonia_sample_ids", None)
        if sample_ids is None:
            return network_kwargs
        if str(getattr(self.model_cfg, "point_encoder", "")) not in {
            "utonia",
            "utonia_cached",
        }:
            return network_kwargs
        feature_encoder = getattr(getattr(self.network, "dit", None), "feature_encoder", None)
        if not hasattr(feature_encoder, "prepare_static_context"):
            raise RuntimeError("sample IDs require a Utonia feature encoder")
        network_kwargs["static_geometry"] = feature_encoder.prepare_static_context(
            network_kwargs["x0"],
            network_kwargs["y"],
            sample_ids=sample_ids,
        )
        return network_kwargs

    @torch.no_grad()
    def sample_candidates(
        self,
        pc_action: torch.Tensor,
        pc_anchor: torch.Tensor,
        num_trials: int = 1,
        seed: Optional[int] = None,
        sample_ids: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Sample goal point clouds without reading a ground-truth goal.

        This is the deployment-facing seam for the reconstructed fixed-frame
        TAX-DPD module.  Each candidate owns an independent deterministic random
        stream when ``seed`` is provided, so asking for more candidates cannot
        change an earlier candidate.  The method intentionally iterates the
        progressive sampler and retains only its final tensors; the historical
        ``p_sample_loop`` stores every diffusion step and ignores caller-provided
        initial noise.

        Args:
            pc_action: observed placed-object points with shape ``[B, N, 3]``.
            pc_anchor: observed support points with shape ``[B, M, 3]``.
            num_trials: number of complete candidate goal point clouds.
            seed: optional base seed; candidate ``i`` uses ``seed + i``.

        Returns:
            Tensor with shape ``[B, K, N, 3]`` in the same centered scene frame
            as the inputs.
        """

        if isinstance(num_trials, bool) or not isinstance(num_trials, int) or num_trials <= 0:
            raise ValueError("num_trials must be a positive integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")
        for name, points in (("pc_action", pc_action), ("pc_anchor", pc_anchor)):
            if not isinstance(points, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if points.ndim != 3 or points.shape[-1] != 3:
                raise ValueError(f"{name} must have shape [B, N, 3]")
            if points.shape[0] <= 0 or points.shape[1] < 4:
                raise ValueError(f"{name} must contain a non-empty batch and at least four points")
            if not torch.isfinite(points).all():
                raise ValueError(f"{name} contains non-finite values")
        if pc_action.shape[0] != pc_anchor.shape[0]:
            raise ValueError("pc_action and pc_anchor batch sizes must match")
        if sample_ids is not None:
            if len(sample_ids) != pc_action.shape[0]:
                raise ValueError("sample_ids length must match pc_action batch")
            sample_ids = [str(value) for value in sample_ids]

        device = self.device
        action = pc_action.to(device)
        anchor = pc_anchor.to(device)
        batch_size, sample_size = action.shape[:2]
        model_kwargs = self._ddpm_model_kwargs({
            "y": anchor.permute(0, 2, 1),
            "x0": action.permute(0, 2, 1),
            **({"_utonia_sample_ids": sample_ids} if sample_ids is not None else {}),
        })
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                torch.cuda.current_device() if device.index is None else int(device.index)
            ]

        was_training = self.training
        self.eval()
        candidates = []
        try:
            for trial_index in range(num_trials):
                candidate_seed = None if seed is None else seed + trial_index
                with torch.random.fork_rng(devices=cuda_devices, enabled=candidate_seed is not None):
                    if candidate_seed is not None:
                        if device.type == "cuda":
                            with torch.cuda.device(cuda_devices[0]):
                                torch.cuda.manual_seed(candidate_seed)
                        else:
                            torch.manual_seed(candidate_seed)
                    noise_r = (
                        torch.randn(batch_size, 3, 1, device=device) * self.noise_scale
                    )
                    noise_s = (
                        torch.randn(batch_size, 3, sample_size, device=device)
                        * self.noise_scale
                    )
                    if bool(self.model_cfg.zero_shape):
                        noise_s = noise_s - noise_s.mean(dim=2, keepdim=True)

                    final = None
                    for output in self.diffusion.p_sample_loop_progressive(
                        self.network,
                        noise_r.shape,
                        noise_s.shape,
                        noise_r=noise_r,
                        noise_s=noise_s,
                        clip_denoised=True,
                        model_kwargs=model_kwargs,
                        progress=False,
                        device=device,
                    ):
                        final = output
                    if final is None:
                        raise RuntimeError("diffusion sampler returned no timesteps")
                    candidate = (final["sample_r"] + final["sample_s"]).permute(0, 2, 1)
                    if candidate.shape != action.shape or not torch.isfinite(candidate).all():
                        raise RuntimeError(
                            "diffusion sampler returned an invalid candidate with shape "
                            f"{tuple(candidate.shape)}"
                        )
                    candidates.append(candidate)
        finally:
            self.train(was_training)
        return torch.stack(candidates, dim=1)

    @torch.no_grad()
    def _predict_wta(self, batch: Dict, num_trials: Optional[int] = None) -> Dict:
        n = num_trials if num_trials is not None else self.num_wta_trials
        gt = batch["pc"].to(self.device)          # [B, N, 3]
        bs = gt.shape[0]
        pred_action = self.sample_candidates(
            batch["pc_action"],
            batch["pc_anchor"],
            num_trials=n,
        )
        gt_exp = gt[:, None].expand_as(pred_action)
        rmse = flow_rmse(
            pred_action.reshape(bs * n, -1, 3),
            gt_exp.reshape(bs * n, -1, 3),
            mask=False,
            seg=None,
        ).reshape(bs, n)
        cos_sim = flow_cos_sim(
            pred_action.reshape(bs * n, -1, 3),
            gt_exp.reshape(bs * n, -1, 3),
            mask=False,
            seg=None,
        ).reshape(bs, n)
        winner  = torch.argmin(rmse, dim=-1)

        return {
            "pred_actions_wta": pred_action[torch.arange(bs), winner],
            "pred_action":      pred_action,
            "rmse_wta":         rmse[torch.arange(bs), winner],
            "rmse":             rmse,
            "cos_sim_wta":      cos_sim[torch.arange(bs), winner],
            "cos_sim":          cos_sim,
        }


class TAX3Dv2FixedFrameFlowMatchingModule(_TAX3Dv2BaseModule):
    """Controlled Euclidean FM ablation using the fixed-frame TAX-DPD DiT.

    Conditioning, frame/shape decomposition, point slots, and downstream WTA
    result shape match ``TAX3Dv2FixedFrameModule``.  Only the DDPM corruption,
    learned variance, loss, and sampler are replaced.  Rotation corruption is
    intentionally rejected until a path-consistent velocity target exists.
    """

    def __init__(self, network: TAX3Dv2Network, cfg: omegaconf.DictConfig):
        super().__init__(network, cfg)
        if bool(cfg.model.learn_sigma):
            raise ValueError("flow matching requires learn_sigma=false (3-channel velocity heads)")
        if cfg.model.diff_rotation_noise_scale not in (False, 0, 0.0, None):
            raise ValueError("standard Euclidean FM requires rotation corruption to be disabled")
        self.fm_steps = int(cfg.model.fm_steps)
        self.fm_solver = str(cfg.model.fm_solver)
        self.fm_time_scale = float(cfg.model.fm_time_scale)
        # Unlike the DDPM module, FM's source prior is part of the model
        # contract.  Keep it in the resolved config so checkpoint reloads do
        # not silently fall back to the historical ``diff_noise_scale``.
        self.noise_scale = float(
            getattr(cfg.model, "fm_noise_scale", self.noise_scale)
        )
        if self.fm_steps <= 0:
            raise ValueError("fm_steps must be positive")
        if self.fm_solver not in {"euler", "heun"}:
            raise ValueError("fm_solver must be euler or heun")
        if not np.isfinite(self.fm_time_scale) or self.fm_time_scale <= 0:
            raise ValueError("fm_time_scale must be finite and positive")

    def _get_x_start(self, batch: Dict) -> torch.Tensor:
        return batch["pc"].to(self.device).permute(0, 2, 1)

    @staticmethod
    def _split_target(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame = target.mean(dim=2, keepdim=True)
        return frame, target - frame

    def _source(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame, shape = self._split_target(target)
        source_frame = torch.randn_like(frame) * self.noise_scale
        source_shape = torch.randn_like(shape) * self.noise_scale
        if self.model_cfg.zero_shape:
            source_shape = source_shape - source_shape.mean(dim=2, keepdim=True)
        return source_frame, source_shape

    def _source_from_observation(self, pc_action: torch.Tensor) -> torch.Tensor:
        """Build the deployment prior using observation shape only."""
        if pc_action.ndim != 3 or pc_action.shape[1] != 3:
            raise ValueError("pc_action must have shape [B, 3, N]")
        source_frame = torch.randn(
            pc_action.shape[0], 3, 1, device=pc_action.device, dtype=pc_action.dtype
        ) * self.noise_scale
        source_shape = torch.randn_like(pc_action) * self.noise_scale
        if self.model_cfg.zero_shape:
            source_shape = source_shape - source_shape.mean(dim=2, keepdim=True)
        return torch.cat((source_frame, source_shape), dim=2)

    def _velocity_model(self, model_kwargs: Dict[str, torch.Tensor]):
        network_kwargs = dict(model_kwargs)
        sample_ids = network_kwargs.pop("_utonia_sample_ids", None)
        static_geometry = None
        feature_encoder = getattr(getattr(self.network, "dit", None), "feature_encoder", None)
        if (
            str(getattr(self.model_cfg, "point_encoder", ""))
            in {"utonia", "utonia_cached"}
            and hasattr(feature_encoder, "prepare_static_context")
        ):
            if sample_ids is None:
                static_geometry = feature_encoder.prepare_static_context(
                    network_kwargs["x0"], network_kwargs["y"]
                )
            else:
                static_geometry = feature_encoder.prepare_static_context(
                    network_kwargs["x0"], network_kwargs["y"], sample_ids=sample_ids
                )

        def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            frame = state[:, :, :1]
            shape = state[:, :, 1:]
            frame_velocity, shape_velocity = self.network(
                frame,
                shape,
                time * self.fm_time_scale,
                static_geometry=static_geometry,
                **network_kwargs,
            )
            if frame_velocity.shape[1] != 3 or shape_velocity.shape[1] != 3:
                raise ValueError("FM DiT must return exactly three velocity channels per head")
            return torch.cat((frame_velocity, shape_velocity), dim=2)

        return velocity

    @torch.no_grad()
    def sample_candidates(
        self,
        pc_action: torch.Tensor,
        pc_anchor: torch.Tensor,
        num_trials: int = 1,
        seed: Optional[int] = None,
        sample_ids: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Sample target-free goal point clouds with the FM ODE.

        Inputs and outputs use the centered TAX-DPD numeric frame.  Every
        candidate owns an independent RNG stream (``seed + i``), matching the
        fixed-frame DDPM seam and making a request for more candidates prefix
        stable.  No target point cloud is consulted here.
        """

        if (
            isinstance(num_trials, bool)
            or not isinstance(num_trials, int)
            or num_trials <= 0
        ):
            raise ValueError("num_trials must be a positive integer")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        for name, points in (("pc_action", pc_action), ("pc_anchor", pc_anchor)):
            if not isinstance(points, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if points.ndim != 3 or points.shape[-1] != 3:
                raise ValueError(f"{name} must have shape [B, N, 3]")
            if points.shape[0] <= 0 or points.shape[1] < 4:
                raise ValueError(
                    f"{name} must contain a non-empty batch and at least four points"
                )
            if not torch.isfinite(points).all():
                raise ValueError(f"{name} contains non-finite values")
        if pc_action.shape[0] != pc_anchor.shape[0]:
            raise ValueError("pc_action and pc_anchor batch sizes must match")

        device = self.device
        action = pc_action.to(device)
        anchor = pc_anchor.to(device)
        batch_size, sample_size = action.shape[:2]
        model_kwargs = {
            "y": anchor.permute(0, 2, 1),
            "x0": action.permute(0, 2, 1),
        }
        if sample_ids is not None:
            if len(sample_ids) != batch_size:
                raise ValueError("sample_ids length must match pc_action batch")
            model_kwargs["_utonia_sample_ids"] = [str(value) for value in sample_ids]
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            ]

        was_training = self.training
        self.eval()
        candidates = []
        try:
            for trial_index in range(num_trials):
                candidate_seed = None if seed is None else seed + trial_index
                with torch.random.fork_rng(
                    devices=cuda_devices, enabled=candidate_seed is not None
                ):
                    if candidate_seed is not None:
                        if device.type == "cuda":
                            with torch.cuda.device(cuda_devices[0]):
                                torch.cuda.manual_seed(candidate_seed)
                        else:
                            torch.manual_seed(candidate_seed)
                    source = self._source_from_observation(
                        action.permute(0, 2, 1)
                    )
                    final = integrate_velocity(
                        self._velocity_model(model_kwargs),
                        source,
                        steps=self.fm_steps,
                        method=self.fm_solver,
                    )
                    # The first state slot is the frame centroid; the
                    # remaining N slots are zero-mean shape residuals.
                    # Reconstruct the N-point goal before returning it.
                    candidate = (
                        final[:, :, :1] + final[:, :, 1:]
                    ).permute(0, 2, 1)
                    if (
                        candidate.shape != action.shape
                        or not torch.isfinite(candidate).all()
                    ):
                        raise RuntimeError(
                            "FM sampler returned an invalid candidate with shape "
                            f"{tuple(candidate.shape)}"
                        )
                    candidates.append(candidate)
        finally:
            self.train(was_training)
        return torch.stack(candidates, dim=1)

    def training_step(self, batch, batch_idx):
        del batch_idx
        self.train()
        target = self._get_x_start(batch)
        target_frame, target_shape = self._split_target(target)
        source_frame, source_shape = self._source(target)
        source = torch.cat((source_frame, source_shape), dim=2)
        endpoint = torch.cat((target_frame, target_shape), dim=2)
        losses, _, _ = flow_matching_loss(
            self._velocity_model(self._model_kwargs(batch)),
            endpoint,
            source=source,
        )
        loss = losses.mean()
        self.log("train/loss", loss, prog_bar=True, add_dataloader_idx=False)
        return loss

    @torch.no_grad()
    def _predict_wta(self, batch: Dict, num_trials: Optional[int] = None) -> Dict:
        n = num_trials if num_trials is not None else self.num_wta_trials
        gt = batch["pc"].to(self.device)
        bs = gt.shape[0]
        pc_action = expand_pcd(batch["pc_action"].to(self.device), n)
        pc_anchor = expand_pcd(batch["pc_anchor"].to(self.device), n)
        gt_exp = expand_pcd(gt, n)
        # Deployment has no target point cloud.  Use only the observed action
        # shape to determine the latent tensor size; never inspect ``batch[pc]``
        # when constructing the FM source state.
        observed_shape = pc_action.permute(0, 2, 1)
        source = self._source_from_observation(observed_shape)
        model_kwargs = {
            "y": pc_anchor.permute(0, 2, 1),
            "x0": pc_action.permute(0, 2, 1),
        }
        final = integrate_velocity(
            self._velocity_model(model_kwargs),
            source,
            steps=self.fm_steps,
            method=self.fm_solver,
        )
        pred_action = (final[:, :, :1] + final[:, :, 1:]).permute(0, 2, 1)
        rmse = flow_rmse(pred_action, gt_exp, mask=False, seg=None).reshape(bs, n)
        cos_sim = flow_cos_sim(pred_action, gt_exp, mask=False, seg=None).reshape(bs, n)
        pred_action = pred_action.reshape(bs, n, -1, 3)
        winner = torch.argmin(rmse, dim=-1)
        return {
            "pred_actions_wta": pred_action[torch.arange(bs, device=self.device), winner],
            "pred_action": pred_action,
            "rmse_wta": rmse[torch.arange(bs, device=self.device), winner],
            "rmse": rmse,
            "cos_sim_wta": cos_sim[torch.arange(bs, device=self.device), winner],
            "cos_sim": cos_sim,
        }


# ─── Mu-frame module ──────────────────────────────────────────────────────────

class TAX3Dv2MuFrameModule(_TAX3Dv2BaseModule):
    """
    TAX3Dv2 training module for the "mu" prediction frame.

    At training time the frame is simulated as a *noisy* perturbation of the
    true goal centroid (batch["noisy_goal"], produced by RPDiffDataset when
    dataset.pred_frame == "noisy_goal").  The goal is centred around this
    noisy frame before being passed to the diffusion.
    At inference the frame should come from the GMM stage; for now we use
    the zero vector (no-op shift) when noisy_goal is absent.
    """

    def __init__(self, network: TAX3Dv2Network, cfg: omegaconf.DictConfig):
        super().__init__(network, cfg)
        self.diffusion = create_diffusion_mu(
            timestep_respacing=None,
            noise_schedule=cfg.model.diff_noise_schedule,
            learn_sigma=cfg.model.learn_sigma,
            diffusion_steps=cfg.model.diff_train_steps,
            # Passed through verbatim; see the fixed-frame module for why.
            time_based_weighting=cfg.model.time_based_weighting,
            rotation_noise_scale=cfg.model.diff_rotation_noise_scale,
        )

    def _get_x_start(self, batch: Dict) -> torch.Tensor:
        pc_goal    = batch["pc"].to(self.device).permute(0, 2, 1)   # [B, 3, N]
        noisy_goal = batch["noisy_goal"].to(self.device).unsqueeze(-1)  # [B, 3, 1]
        # Express the goal relative to the noisy GMM frame estimate
        return pc_goal - noisy_goal

    # NOTE: no _model_kwargs override here. The diffusion splats model_kwargs
    # straight into the DiT (model(xr_t, xs_t, t, **model_kwargs)), so it may
    # only carry keys the DiT signature accepts: y / x0 / rel_pos /
    # finetune_frame. The frame estimate reaches the model a different way --
    # it seeds noise_r in p_sample_loop, which does pred_ref = noise_r.clone().

    @torch.no_grad()
    def _predict_wta(self, batch: Dict, num_trials: Optional[int] = None) -> Dict:
        n = num_trials if num_trials is not None else self.num_wta_trials
        gt = batch["pc"].to(self.device)          # [B, N, 3]
        bs = gt.shape[0]
        sample_size = gt.shape[1]

        pc_action = expand_pcd(batch["pc_action"].to(self.device), n)
        pc_anchor = expand_pcd(batch["pc_anchor"].to(self.device), n)
        gt_exp    = expand_pcd(gt, n)

        has_goal = "noisy_goal" in batch
        if has_goal:
            noisy_goal = expand_pcd(
                batch["noisy_goal"].to(self.device).unsqueeze(1), n
            ).squeeze(1)   # [B*n, 3]
        else:
            noisy_goal = torch.zeros(bs * n, 3, device=self.device)

        # Only DiT-signature keys here -- the diffusion splats this dict
        # straight into model(xr_t, xs_t, t, **model_kwargs).
        model_kwargs = {
            "y":  pc_anchor.permute(0, 2, 1),
            "x0": pc_action.permute(0, 2, 1),
        }

        noise_r = torch.randn(bs * n, 3, 1,           device=self.device) * self.noise_scale
        noise_s = torch.randn(bs * n, 3, sample_size, device=self.device) * self.noise_scale

        # SpacedDiffusionMuFrame.p_sample_loop returns (final_dict, results)
        # final_dict = {"sample_r": [B*n, 3, 1], "sample_s": [B*n, 3, N], "prev_sample_r": ...}
        final_dict, _ = self.diffusion.p_sample_loop(
            self.network,
            noise_r.shape,
            noise_s.shape,
            noise_r=noise_r,
            noise_s=noise_s,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=self.device,
        )

        # Reconstruct in scene frame: add back the noisy-goal frame offset
        ng = noisy_goal.unsqueeze(-1)    # [B*n, 3, 1]
        pred_action = (final_dict["sample_r"] + final_dict["sample_s"] + ng).permute(0, 2, 1)

        rmse    = flow_rmse(pred_action,    gt_exp, mask=False, seg=None).reshape(bs, n)
        cos_sim = flow_cos_sim(pred_action, gt_exp, mask=False, seg=None).reshape(bs, n)
        pred_action = pred_action.reshape(bs, n, -1, 3)
        winner  = torch.argmin(rmse, dim=-1)

        return {
            "pred_actions_wta": pred_action[torch.arange(bs), winner],
            "pred_action":      pred_action,
            "rmse_wta":         rmse[torch.arange(bs), winner],
            "rmse":             rmse,
            "cos_sim_wta":      cos_sim[torch.arange(bs), winner],
            "cos_sim":          cos_sim,
        }
