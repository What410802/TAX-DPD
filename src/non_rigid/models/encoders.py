from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import numpy as np

from functools import partial

#################################################################################
#                               Point Cloud Encoders                            #
#################################################################################

def mlp_encoder(in_channels, out_channels):
    """
    MLP encoder for point clouds.
    """
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )

def pn2_encoder(in_channels, out_channels, model_cfg):
    """
    PointNet++ encoder for point clouds.
    """
    # Keep PointNet++ dependencies lazy so the optional Utonia backend can be
    # imported in its standalone CUDA image without rpad/torch-geometric.
    import torch_geometric.data as tgd
    from non_rigid.nets.pn2 import PN2Dense, PN2DenseParams

    pn2_params = PN2DenseParams()
    # if model_cfg.object_scale is not None or model_cfg.scene_scale is not None:
    if model_cfg.object_scale is not None:
        pn2_params.sa1.r = 0.5 * model_cfg.pcd_scale
        pn2_params.sa2.r = 1.0 * model_cfg.pcd_scale
    else:
        pn2_params.sa1.r = 0.2 * model_cfg.pcd_scale
        pn2_params.sa2.r = 0.4 * model_cfg.pcd_scale

    class PN2DenseWrapper(nn.Module):
        def __init__(self, in_channels, out_channels, p):
            super().__init__()
            self.pn2dense = PN2Dense(
                in_channels=in_channels - 3,
                out_channels=out_channels,
                p=p,
            )

        def forward(self, x):
            batch_size, num_channels = x.shape[0], x.shape[1]
            batch_indices = torch.arange(
                batch_size, device=x.device
            ).repeat_interleave(x.shape[2])

            if num_channels == 3:
                input_batch = tgd.Batch(
                    pos=x.permute(0, 2, 1).reshape(-1, 3), batch=batch_indices
                )
            elif num_channels > 3:
                input_batch = tgd.Batch(
                    pos=x[:, :3, :].permute(0, 2, 1).reshape(-1, 3),
                    x=x[:, 3:, :].permute(0, 2, 1).reshape(-1, num_channels - 3),
                    batch=batch_indices,
                )
            else:
                raise ValueError(f"Invalid number of input channels: {num_channels}")
            
            output = self.pn2dense(input_batch)
            output = output.reshape(batch_size, -1, output.shape[-1]).permute(0, 2, 1)
            return output
    
    return PN2DenseWrapper(in_channels=in_channels, out_channels=out_channels, p=pn2_params)


class UtoniaSlotFeatureEncoder(nn.Module):
    """Frozen Utonia encoder with an ordered-slot output contract.

    Utonia is an optional runtime dependency and is imported only when this
    backend is selected.  Coordinates are converted from TAX-DPD's numeric
    frame back to metres before Utonia's documented transform; missing color
    and normal modalities are represented by zeros.  The backbone is frozen
    and a small trainable 1x1 projection maps its 576 channels to DiT width.
    """

    def __init__(
        self,
        hidden_size,
        model_cfg,
        *,
        backbone: Optional[nn.Module] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.input_scale_factor = float(
            getattr(model_cfg, "utonia_input_scale_factor", 15.0)
        )
        self.transform_scale = float(getattr(model_cfg, "utonia_transform_scale", 0.5))
        self.center_shift = bool(getattr(model_cfg, "utonia_center_shift", False))
        self.transform_seed = int(getattr(model_cfg, "utonia_transform_seed", 1701))
        if not np.isfinite(self.input_scale_factor) or self.input_scale_factor <= 0:
            raise ValueError("utonia_input_scale_factor must be finite and positive")
        if not np.isfinite(self.transform_scale) or self.transform_scale <= 0:
            raise ValueError("utonia_transform_scale must be finite and positive")

        self._external_transform = transform
        if backbone is None:
            checkpoint = getattr(model_cfg, "utonia_checkpoint", None)
            if not checkpoint:
                raise ValueError(
                    "point_encoder=utonia requires model.utonia_checkpoint"
                )
            try:
                import utonia
            except ImportError as error:
                raise ImportError(
                    "point_encoder=utonia requires the Utonia runtime; "
                    "use the B300 image or install its CUDA extensions"
                ) from error
            custom_config = dict(
                enc_patch_size=[1024 for _ in range(5)],
                enable_flash=bool(getattr(model_cfg, "utonia_enable_flash", False)),
                shuffle_orders=False,
            )
            backbone = utonia.load(str(Path(checkpoint)), custom_config=custom_config)
            if transform is None:
                transform_config = [
                    dict(type="RandomScale", scale=[self.transform_scale, self.transform_scale]),
                ]
                if self.center_shift:
                    transform_config.append(dict(type="CenterShift", apply_z=True))
                transform_config.extend(
                    [
                        dict(
                            type="GridSample",
                            grid_size=0.01,
                            hash_type="fnv",
                            mode="train",
                            return_grid_coord=True,
                            return_inverse=True,
                        ),
                        dict(type="NormalizeColor"),
                        dict(type="ToTensor"),
                        dict(
                            type="Collect",
                            keys=("coord", "grid_coord", "color", "inverse"),
                            feat_keys=("coord", "color", "normal"),
                        ),
                    ]
                )
                transform = utonia.transform.Compose(transform_config)
                self._external_transform = transform
        self.backbone = backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.projection = nn.Conv1d(576, hidden_size, kernel_size=1)

    def train(self, mode: bool = True):
        super().train(mode)
        # Parent ``model.train()`` must not re-enable Utonia drop path.
        self.backbone.eval()
        return self

    def _transform(self, coordinates):
        if self._external_transform is None:
            raise RuntimeError("Utonia transform is not configured")
        # Importing these helpers lazily keeps the baseline importable without
        # Utonia's CUDA extensions.
        from non_rigid.models.utonia_slot_adapter import build_utonia_input

        points = build_utonia_input(
            (coordinates.detach().float().cpu().numpy() / self.input_scale_factor)
            .astype(np.float32, copy=False)
        )
        # Utonia's documented train-mode GridSample randomly chooses a point
        # per voxel.  Use an isolated fixed stream so the frozen condition is
        # repeatable without perturbing the caller's NumPy RNG.
        state = np.random.get_state()
        try:
            np.random.seed(self.transform_seed)
            return self._external_transform(points)
        finally:
            np.random.set_state(state)

    def frozen_features(self, x):
        if x.ndim != 3 or x.shape[1] < 3 or x.shape[2] < 4:
            raise ValueError("Utonia slot encoder expects [B, C>=3, N>=4]")
        outputs = []
        for coordinates in x[:, :3, :].permute(0, 2, 1):
            point = self._transform(coordinates)
            for key, value in tuple(point.items()):
                if isinstance(value, torch.Tensor):
                    point[key] = value.to(device=x.device, non_blocking=True)
            with torch.inference_mode():
                encoded = self.backbone(point)
                from non_rigid.models.utonia_slot_adapter import upcast_slot_features

                slot_features = upcast_slot_features(encoded)
            if slot_features.shape[0] != x.shape[2] or slot_features.shape[1] != 576:
                raise ValueError(
                    "Utonia must return one 576-D feature per original point slot"
                )
            outputs.append(slot_features)
        return torch.stack(outputs, dim=0).permute(0, 2, 1)

    def forward(self, x):
        return self.projection(self.frozen_features(x))


class UtoniaJointFeatureEncoder(nn.Module):
    """TAX3Dv2 joint encoder using shared frozen Utonia geometry features.

    Static action+anchor geometry is encoded once as a joint Utonia scene.
    Dynamic FM state/shape/flow channels stay in small MLPs, so Heun sampling
    does not rerun the 137M frozen backbone at every function evaluation.
    """

    def __init__(self, in_channels, hidden_size, model_cfg, *, slot_encoder=None):
        super().__init__()
        del in_channels
        self.model_cfg = model_cfg
        self.hidden_size = hidden_size
        self.geometry_encoder = (
            slot_encoder
            if slot_encoder is not None
            else UtoniaSlotFeatureEncoder(hidden_size, model_cfg)
        )
        self.dynamic_encoder = mlp_encoder(3, hidden_size)
        self.flow_feature_encoder = mlp_encoder(9, hidden_size)
        self.action_mixer = mlp_encoder(3 * hidden_size, hidden_size)
        self.action_role = nn.Parameter(torch.zeros(1, hidden_size, 1))
        self.anchor_role = nn.Parameter(torch.zeros(1, hidden_size, 1))

    def prepare_static_context(self, x0, y):
        action_size = x0.shape[-1]
        scene = self.geometry_encoder(torch.cat((x0, y), dim=2))
        return (
            scene[:, :, :action_size] + self.action_role,
            scene[:, :, action_size:] + self.anchor_role,
        )

    def forward(self, x, y, x0, static_geometry=None):
        if self.model_cfg.type == "flow":
            x_flow = x
            x_recon = x + x0
        else:
            x_flow = x - x0
            x_recon = x
        if static_geometry is None:
            action_enc, anchor_pred_enc = self.prepare_static_context(x0, y)
        else:
            if not isinstance(static_geometry, tuple) or len(static_geometry) != 2:
                raise ValueError("static_geometry must be an (action, anchor) tuple")
            action_enc, anchor_pred_enc = static_geometry
        pred_action_enc = self.dynamic_encoder(x_recon)

        shape = x_recon - torch.mean(x_recon, dim=2, keepdim=True)
        flow_zeromean = x_flow - torch.mean(x_flow, dim=2, keepdim=True)
        feature_enc = self.flow_feature_encoder(
            torch.cat([shape, x_flow, flow_zeromean], dim=1)
        )
        x_enc = self.action_mixer(
            torch.cat([action_enc, pred_action_enc, feature_enc], dim=1)
        ).permute(0, 2, 1)
        return x_enc, anchor_pred_enc.permute(0, 2, 1)


class UtoniaDisjointFeatureEncoder(nn.Module):
    """Disjoint counterpart retained for non-joint TAX3Dv2 variants."""

    def __init__(self, in_channels, hidden_size, model_cfg, *, slot_encoder=None):
        super().__init__()
        del in_channels
        self.model_cfg = model_cfg
        self.hidden_size = hidden_size
        self.geometry_encoder = (
            slot_encoder
            if slot_encoder is not None
            else UtoniaSlotFeatureEncoder(hidden_size, model_cfg)
        )
        self.shape_encoder = mlp_encoder(3, hidden_size)
        self.flow_zeromean_encoder = mlp_encoder(3, hidden_size)
        self.x_corr_encoder = mlp_encoder(3, hidden_size)
        self.action_mixer = mlp_encoder(5 * hidden_size, hidden_size)

    def forward(self, x, y, x0):
        if self.model_cfg.type == "flow":
            x_flow = x
            x_recon = x + x0
        else:
            x_flow = x - x0
            x_recon = x
        x_enc = self.geometry_encoder(x)
        x0_enc = self.geometry_encoder(x0)
        y_enc = self.geometry_encoder(y).permute(0, 2, 1)
        shape_enc = self.shape_encoder(x_recon - x_recon.mean(dim=2, keepdim=True))
        flow_zeromean_enc = self.flow_zeromean_encoder(
            x_flow - x_flow.mean(dim=2, keepdim=True)
        )
        x_corr_enc = self.x_corr_encoder(x_recon if self.model_cfg.type == "flow" else x_flow)
        x_enc = self.action_mixer(
            torch.cat([x_enc, x0_enc, shape_enc, flow_zeromean_enc, x_corr_enc], dim=1)
        ).permute(0, 2, 1)
        return x_enc, y_enc

#################################################################################
#                                 Feature Encoders                              #
#################################################################################

class DisjointFeatureEncoder(nn.Module):
    """
    TODO: fill this out
    """
    def __init__(self, in_channels, hidden_size, model_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.model_cfg = model_cfg

        if self.model_cfg.point_encoder == "utonia":
            self._utonia_encoder = UtoniaDisjointFeatureEncoder(
                in_channels, hidden_size, model_cfg
            )
            return

        # Initializing point cloud encoder wrapper.
        if self.model_cfg.point_encoder == "mlp":
            encoder_fn = partial(mlp_encoder, in_channels=self.in_channels)
        elif self.model_cfg.point_encoder == "pn2":
            encoder_fn = partial(pn2_encoder, in_channels=self.in_channels, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid point_encoder: {self.model_cfg.point_encoder}")

        # Creating base encoders - action (x0), anchor (y), and noised prediction (x).
        self.x_encoder = encoder_fn(out_channels=hidden_size)
        self.x0_encoder = encoder_fn(out_channels=hidden_size)
        self.y_encoder = encoder_fn(out_channels=hidden_size)

        # Creating extra feature encoders, if necessary.
        if self.model_cfg.feature:
            self.shape_encoder = encoder_fn(out_channels=hidden_size)
            self.flow_zeromean_encoder = encoder_fn(out_channels=hidden_size)
            self.x_corr_encoder = encoder_fn(out_channels=hidden_size)
            self.action_mixer = mlp_encoder(5 * hidden_size, hidden_size)
        else:
            self.action_mixer = mlp_encoder(2 * hidden_size, hidden_size)
    
    def forward(self, x, y, x0):
        """
        TODO: fill this out
        """
        if hasattr(self, "_utonia_encoder"):
            return self._utonia_encoder(x, y, x0)
        if self.model_cfg.type == "flow":
            x_flow = x
            x_recon = x + x0
        else:
            x_flow = x - x0
            x_recon = x
        
        # Encode base features - action (x0), anchor (y), and noised prediction (x).
        x_enc = self.x_encoder(x)
        x0_enc = self.x0_encoder(x0)
        y_enc = self.y_encoder(y).permute(0, 2, 1)

        # Encode extra features, if necessary.
        if self.model_cfg.feature:
            shape_enc = self.shape_encoder(
                x_recon - torch.mean(x_recon, dim=2, keepdim=True)
            )
            flow_zeromean_enc = self.flow_zeromean_encoder(
                x_flow - torch.mean(x_flow, dim=2, keepdim=True)
            )
            x_corr_enc = self.x_corr_encoder(
                x_recon if self.model_cfg.type == "flow" else x_flow
            )
            action_features = [x_enc, x0_enc, shape_enc, flow_zeromean_enc, x_corr_enc]
        else:
            action_features = [x_enc, x0_enc]

        # Compress action features to hidden size through action mixer.
        x_enc = torch.cat(action_features, dim=1)
        x_enc = self.action_mixer(x_enc).permute(0, 2, 1)

        return x_enc, y_enc

class JointFeatureEncoder(nn.Module):
    """
    TODO: fill this out
    """
    def __init__(self, in_channels, hidden_size, model_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.model_cfg = model_cfg

        if self.model_cfg.point_encoder == "utonia":
            self._utonia_encoder = UtoniaJointFeatureEncoder(
                in_channels, hidden_size, model_cfg
            )
            return

        # Initializing point cloud encoder wrapper.
        if self.model_cfg.point_encoder == "mlp":
            encoder_fn = partial(mlp_encoder, in_channels=self.in_channels)
        elif self.model_cfg.point_encoder == "pn2":
            encoder_fn = partial(pn2_encoder, in_channels=self.in_channels, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid point_encoder: {self.model_cfg.point_encoder}")

        # Creating base encoders - action-frame, and prediction-frame.
        self.action_encoder = encoder_fn(out_channels=hidden_size)
        if self.model_cfg.one_hot_recon:
            self.pred_encoder = encoder_fn(in_channels=self.in_channels + 1, out_channels=hidden_size)
        else:
            self.pred_encoder = encoder_fn(out_channels=hidden_size)

        # Creating extra feature encoders, if necessary.
        if self.model_cfg.feature:
            self.feature_encoder = encoder_fn(in_channels=9, out_channels=hidden_size)
            self.action_mixer = mlp_encoder(3 * hidden_size, hidden_size)
        else:
            self.action_mixer = mlp_encoder(2 * hidden_size, hidden_size)
    
    def prepare_static_context(self, x0, y):
        if not hasattr(self, "_utonia_encoder"):
            raise RuntimeError("static context is available only for Utonia")
        return self._utonia_encoder.prepare_static_context(x0, y)

    def forward(self, x, y, x0, static_geometry=None):
        """
        TODO: fill this out
        """
        if hasattr(self, "_utonia_encoder"):
            return self._utonia_encoder(
                x, y, x0, static_geometry=static_geometry
            )
        if static_geometry is not None:
            raise ValueError("static_geometry is unsupported by this point encoder")
        if self.model_cfg.type == "flow":
            x_flow = x
            x_recon = x + x0
        else:
            x_flow = x - x0
            x_recon = x
        
        # Encode base features - action-frame, and prediction frame.
        action_size = x0.shape[-1]
        action_enc = self.action_encoder(x0)
        if self.model_cfg.one_hot_recon:
            x_recon_one_hot = torch.cat([x_recon, torch.ones_like(x_recon[:, :1, :])], dim=1)
            y_one_hot = torch.cat([y, torch.zeros_like(y[:, :1, :])], dim=1)
            pred_enc = self.pred_encoder(torch.cat([x_recon_one_hot, y_one_hot], dim=-1))
        else:
            pred_enc = self.pred_encoder(torch.cat([x_recon, y], dim=-1))
        action_pred_enc, anchor_pred_enc = pred_enc[:, :, :action_size], pred_enc[:, :, action_size:]
        anchor_pred_enc = anchor_pred_enc.permute(0, 2, 1)

        # Encode extra features, if necessary.
        if self.model_cfg.feature:
            shape = x_recon - torch.mean(x_recon, dim=2, keepdim=True)
            flow_zeromean = x_flow - torch.mean(x_flow, dim=2, keepdim=True)
            feature_enc = self.feature_encoder(
                torch.cat([shape, x_flow, flow_zeromean], dim=1)
            )
            action_features = [action_enc, action_pred_enc, feature_enc]
        else:
            action_features = [action_enc, action_pred_enc]
        
        # Compress action features to hidden size through action mixer.
        x_enc = torch.cat(action_features, dim=1)
        x_enc = self.action_mixer(x_enc).permute(0, 2, 1)

        return x_enc, anchor_pred_enc
