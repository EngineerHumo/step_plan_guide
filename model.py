import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PromptEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossAttentionFusion(nn.Module):
    """Cross attention between prompt features (query) and image features (key/value)."""

    def __init__(self, channels: int):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        # A prompt skip connection ensures prompt-dependent signals survive even
        # if the attention branch is still warming up.
        self.prompt_skip = nn.Conv2d(channels, channels, kernel_size=1)
        # Start with full-strength attention to keep prompts influential from
        # the first step instead of relying on a tiny residual gate.
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, prompt_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            prompt_feat: (B, C, H, W)
            image_feat:  (B, C, H, W)
        Attention flatten shapes:
            Q: (B, H*W, C)
            K/V: (B, H*W, C)
            attn: (B, H*W, H*W)
            attn_out: (B, H*W, C) -> reshaped back to (B, C, H, W)
        """
        b, c, h, w = prompt_feat.shape
        q = self.q_proj(prompt_feat).flatten(2).transpose(1, 2)  # (B, H*W, C)
        k = self.k_proj(image_feat).flatten(2).transpose(1, 2)  # (B, H*W, C)
        v = self.v_proj(image_feat).flatten(2).transpose(1, 2)  # (B, H*W, C)

        attn_scores = torch.matmul(q, k.transpose(1, 2)) / (c ** 0.5)  # (B, H*W, H*W)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # (B, H*W, C)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)

        prompt_residual = self.prompt_skip(prompt_feat)
        return image_feat + prompt_residual + self.gamma * attn_out


class SpatialMLPBlock(nn.Module):
    """MLP block that mixes global spatial context and channel information."""

    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        hidden_dim = channels * expansion
        self.norm = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        # Spatial gating is driven by global context to propagate long-range signals.
        self.spatial_gate = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        tokens = self.norm(tokens)

        # Channel mixing on every token.
        mixed = self.channel_mlp(tokens)

        # Global spatial mixing: summarize all tokens and gate the mixed output.
        global_context = tokens.mean(dim=1)  # (B, C)
        gate = torch.sigmoid(self.spatial_gate(global_context)).unsqueeze(1)  # (B, 1, C)
        mixed = mixed * gate

        mixed = mixed.transpose(1, 2).reshape(b, c, h, w)
        return x + mixed


class SpatialMLPRefiner(nn.Module):
    """Stack of spatial MLP blocks to model long-range dependencies without ViT."""

    def __init__(self, channels: int, depth: int = 3, expansion: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([SpatialMLPBlock(channels, expansion) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        self.up4 = UpBlock(encoder_channels[4], encoder_channels[3], encoder_channels[3])
        self.up3 = UpBlock(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        self.up2 = UpBlock(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        self.up1 = UpBlock(encoder_channels[1], encoder_channels[0], encoder_channels[0])
        self.final_conv = nn.Conv2d(encoder_channels[0], 1, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4, c5 = features
        x = self.up4(c5, c4)
        x = self.up3(x, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)
        return self.final_conv(x)


class PRPSegmenter(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.prompt_encoder = PromptEncoder(in_channels=1, out_channels=512)
        self.attention = CrossAttentionFusion(channels=512)
        self.mlp_refiner = SpatialMLPRefiner(channels=512, depth=3, expansion=4)

        self.decoder = UNetDecoder(
            encoder_channels=[64, 64, 128, 256, 512]
        )

    def forward(self, image: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.bn1(self.conv1(image)))  # (B, 64, H/2, W/2)
        x2 = self.layer1(self.maxpool(x1))           # (B, 64, H/4, W/4)
        x3 = self.layer2(x2)                         # (B, 128, H/8, W/8)
        x4 = self.layer3(x3)                         # (B, 256, H/16, W/16)
        x5 = self.layer4(x4)                         # (B, 512, H/32, W/32)

        prompt_feat = self.prompt_encoder(heatmap)
        prompt_feat = F.interpolate(prompt_feat, size=x5.shape[2:], mode="bilinear", align_corners=False)

        fused = self.attention(prompt_feat, x5)
        fused = self.mlp_refiner(fused)

        logits = self.decoder([x1, x2, x3, x4, fused])
        logits = F.interpolate(logits, size=image.shape[2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)
