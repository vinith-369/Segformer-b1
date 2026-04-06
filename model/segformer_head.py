
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Linear Embedding: flatten spatial dims -> linear projection."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """SegFormer all-MLP decoder head.

    Takes multi-scale features from MiT encoder (C1-C4) at strides [4, 8, 16, 32],
    projects each to a common embedding_dim via MLP, upsamples to C1's resolution,
    concatenates, fuses through Conv-BN-ReLU, and predicts per-pixel classes.

    Args:
        in_channels: List of channel dims from encoder stages, e.g. [64, 128, 320, 512].
        embedding_dim: Unified channel dim for all MLP projections.
        num_classes: Number of segmentation classes (19 for Cityscapes).
        dropout_ratio: Dropout probability before final prediction.
    """

    def __init__(self, in_channels, embedding_dim=256, num_classes=19, dropout_ratio=0.1):
        super().__init__()
        assert len(in_channels) == 4

        c1_in, c2_in, c3_in, c4_in = in_channels

        # MLP projections for each scale
        self.linear_c4 = MLP(input_dim=c4_in, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in, embed_dim=embedding_dim)

        # Fuse concatenated features: Conv1x1 + BN + ReLU
        # (replaces mmcv's ConvModule with SyncBN)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: list of 4 tensors [C1, C2, C3, C4] from encoder,
                      at resolutions [H/4, H/8, H/16, H/32].
        Returns:
            Logits at C1 resolution (H/4 x W/4).
        """
        c1, c2, c3, c4 = features
        n = c4.shape[0]

        # Project each scale to embedding_dim, reshape back to spatial
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # Concatenate and fuse
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x
