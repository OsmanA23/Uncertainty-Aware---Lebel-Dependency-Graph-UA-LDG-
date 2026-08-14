"""
models/backbone.py
──────────────────
DenseNet-121 visual encoder.

Supports two pretrained weight sources:
  "imagenet"        — standard torchvision ImageNet weights (default)
  "torchxrayvision" — weights pretrained on MIMIC-CXR + other CXR datasets
                      via TorchXRayVision (Cohen et al., MICCAI 2022).
                      Provides substantially better initialisation for CXR
                      classification than ImageNet weights.

Takes a CXR image (B, 3, H, W) and returns:
  - img_embed : (B, embed_dim)         global image embedding via GAP
  - feat_map  : (B, embed_dim, h, w)   spatial feature map (for future use)
"""

import torch
import torch.nn as nn
from torchvision import models


class CXRBackbone(nn.Module):
    """
    DenseNet-121 backbone for chest X-ray classification.

    Args:
        embed_dim          (int)  : Expected 1024 for DenseNet-121.
        pretrained         (bool) : Load pretrained weights.
        pretrained_source  (str)  : "imagenet" | "torchxrayvision"
        freeze_bn          (bool) : Freeze batch normalisation statistics.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        pretrained: bool = True,
        pretrained_source: str = "imagenet",
        freeze_bn: bool = False,
    ) -> None:
        super().__init__()

        # Always initialise architecture from torchvision
        # (ImageNet weights as starting point, or random if pretrained=False)
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)

        self.features  = densenet.features
        self.gap       = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_dim = embed_dim

        if pretrained and pretrained_source == "torchxrayvision":
            self._load_torchxrayvision_weights()

        if freeze_bn:
            self._freeze_bn()

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 3, H, W)  normalised CXR images

        Returns:
            img_embed : (B, embed_dim)
            feat_map  : (B, embed_dim, h, w)
        """
        feat_map  = self.features(x)          # (B, 1024, h, w)
        feat_map  = torch.relu(feat_map)
        img_embed = self.gap(feat_map)         # (B, 1024, 1, 1)
        img_embed = img_embed.flatten(1)       # (B, 1024)
        return img_embed, feat_map

    # ── TorchXRayVision weight loading ────────────────────────────────────────
    def _load_torchxrayvision_weights(self) -> None:
        """
        DenseNet-121 weights pretrained on MIMIC-CXR (CheXpert labels) via TorchXRayVision.

        Reference:
            Cohen JP et al. "TorchXRayVision: A library of chest X-ray
            datasets and models." MICCAI Workshop, 2022.

        Uses 'densenet121-res224-mimic_ch' — trained on MIMIC-CXR with the
        same CheXpert labelling scheme as our dataset.
        """
        try:
            import torchxrayvision as xrv

            xrv_model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")

            # TorchXRayVision state_dict keys: "features.conv0.weight", ...
            # Our self.features state_dict keys: "conv0.weight", ...
            # Strip the "features." prefix to align.
            xrv_feat_state = {
                k[len("features."):]: v
                for k, v in xrv_model.state_dict().items()
                if k.startswith("features.")
            }

            # TorchXRayVision is trained on grayscale (1-channel) X-rays.
            # Adapt conv0 to 3-channel RGB by repeating and scaling.
            if "conv0.weight" in xrv_feat_state:
                w = xrv_feat_state["conv0.weight"]  # [64, 1, 7, 7]
                if w.shape[1] == 1:
                    xrv_feat_state["conv0.weight"] = w.repeat(1, 3, 1, 1) / 3.0

            result = self.features.load_state_dict(xrv_feat_state, strict=False)
            n_loaded = len(xrv_feat_state) - len(result.missing_keys)
            print(
                f"[backbone] TorchXRayVision weights loaded: "
                f"{n_loaded}/{len(xrv_feat_state)} layers  "
                f"(missing={len(result.missing_keys)}, "
                f"unexpected={len(result.unexpected_keys)})"
            )

        except ImportError:
            print(
                "[backbone] Warning: torchxrayvision not installed. "
                "Falling back to ImageNet weights. "
                "Install with: pip install torchxrayvision"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _freeze_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def get_embed_dim(self) -> int:
        return self.embed_dim
