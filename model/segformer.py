# ---------------------------------------------------------------
# Full SegFormer model = MiT encoder + All-MLP decoder
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mix_transformer import mit_b1
from .segformer_head import SegFormerHead


class SegFormer(nn.Module):
    """SegFormer-B1 for semantic segmentation.

    Args:
        num_classes: Number of output classes (19 for Cityscapes).
        embedding_dim: Decoder embedding dimension (256 by default).
        pretrained_path: Path to pretrained MiT-B1 encoder weights.
    """

    def __init__(self, num_classes=19, embedding_dim=256, pretrained_path=None):
        super().__init__()
        self.num_classes = num_classes

        # Encoder: MiT-B1
        self.encoder = mit_b1()
        self.in_channels = self.encoder.embed_dims  # [64, 128, 320, 512]

        # Load pretrained encoder weights
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # Decoder: all-MLP head
        self.decoder = SegFormerHead(
            in_channels=self.in_channels,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

    def _load_pretrained(self, path):
        """Load pretrained MiT-B1 encoder weights with flexible key handling."""
        state_dict = torch.load(path, map_location='cpu')

        # Handle wrapped state dicts
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Strip common prefixes from community re-uploads
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ['backbone.', 'encoder.']:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v
        state_dict = cleaned

        # Remove classification head keys if present
        state_dict.pop('head.weight', None)
        state_dict.pop('head.bias', None)

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"[SegFormer] Loaded pretrained encoder from: {path}")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    def get_param_groups(self):
        """Return parameter groups for differential learning rates.

        Group 0: Encoder non-norm parameters (base LR)
        Group 1: Encoder norm parameters (base LR, no weight decay)
        Group 2: Decoder parameters (10x base LR)
        """
        encoder_params = []
        encoder_norm_params = []
        decoder_params = []

        for name, param in self.encoder.named_parameters():
            if 'norm' in name:
                encoder_norm_params.append(param)
            else:
                encoder_params.append(param)

        for param in self.decoder.parameters():
            decoder_params.append(param)

        return [encoder_params, encoder_norm_params, decoder_params]

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W].
        Returns:
            Segmentation logits at 1/4 resolution [B, num_classes, H/4, W/4].
        """
        features = self.encoder(x)  # List of 4 multi-scale features
        out = self.decoder(features)
        return out
