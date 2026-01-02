import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PromptEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class BiCrossAttentionFusion(nn.Module):
    """Cross attention where the prompt queries the image features."""

    def __init__(self, channels: int = 512, num_heads: int = 8):
        super().__init__()
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, img_feat: torch.Tensor, prompt_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = img_feat.shape
        img_tokens = img_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        prompt_tokens = prompt_feat.flatten(2).transpose(1, 2)  # (B, HW, C)

        attn_scores = torch.matmul(prompt_tokens, img_tokens.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, img_tokens)

        fused_tokens = img_tokens + attn_out
        fused_img = fused_tokens.transpose(1, 2).reshape(b, c, h, w)
        prompt_out = prompt_feat
        return fused_img, prompt_out


class ViTFeatureRefiner(nn.Module):
    """Lightweight ViT-style encoder to model long-range dependencies."""

    def __init__(self, channels: int, num_layers: int = 1, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @staticmethod
    def _build_2d_sincos_position_embedding(height: int, width: int, channels: int, device, dtype):
        if channels % 4 != 0:
            raise ValueError("Channels for positional embedding must be divisible by 4")
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        omega = torch.arange(channels // 4, device=device, dtype=dtype) / (channels // 4)
        omega = 1.0 / (10000 ** omega)
        out_y = torch.einsum("hw,c->hwc", grid_y, omega)
        out_x = torch.einsum("hw,c->hwc", grid_x, omega)
        pos_emb = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
        pos_emb = pos_emb.reshape(1, height * width, channels)
        return pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        pos_emb = self._build_2d_sincos_position_embedding(h, w, c, x.device, x.dtype)
        tokens = tokens + pos_emb
        encoded = self.encoder(tokens)
        return encoded.transpose(1, 2).reshape(b, c, h, w)


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
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(self, stem_channels: int = 32):
        super().__init__()
        self.up4 = UpBlock(2048, 1024, 1024)
        self.up3 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up1 = UpBlock(256, 64, 64)
        self.up0 = UpBlock(64, stem_channels, stem_channels)
        self.final_conv = nn.Conv2d(stem_channels, 1, kernel_size=1)

    def forward(self, features):
        skip0, c1, c2, c3, c4, c5 = features
        x = self.up4(c5, c4)
        x = self.up3(x, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)
        x = self.up0(x, skip0)
        return self.final_conv(x)


class StemConv(nn.Module):
    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_channels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PRPSegmenter(nn.Module):
    def __init__(self, pretrained: bool = True, stem_channels: int = 32):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.stem_conv = StemConv(out_channels=stem_channels)
        self.prompt_encoder = PromptEncoder(in_channels=1, out_channels=512)
        self.proj_img = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.proj_back = nn.Conv2d(512, 2048, kernel_size=1, bias=False)
        self.vit1 = ViTFeatureRefiner(channels=512, num_layers=1, num_heads=8)
        self.vit2 = ViTFeatureRefiner(channels=512, num_layers=1, num_heads=8)
        self.fusion = BiCrossAttentionFusion(channels=512, num_heads=8)

        self.decoder = UNetDecoder(stem_channels=stem_channels)

    def forward(self, image: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        skip0 = self.stem_conv(image)                # (B, 32, H, W)
        x1 = self.relu(self.bn1(self.conv1(image)))  # (B, 64, H/2, W/2)
        x2 = self.layer1(self.maxpool(x1))           # (B, 256, H/4, W/4)
        x3 = self.layer2(x2)                         # (B, 512, H/8, W/8)
        x4 = self.layer3(x3)                         # (B, 1024, H/16, W/16)
        x5 = self.layer4(x4)                         # (B, 2048, H/32, W/32)

        x5p = self.proj_img(x5)                      # (B, 512, H/32, W/32)
        x = self.vit1(x5p)

        has_click = heatmap.flatten(1).max(dim=1).values > 0

        if has_click.any():
            idx = has_click.nonzero(as_tuple=True)[0]
            heatmap_sel = heatmap[idx]
            prompt_feat = self.prompt_encoder(heatmap_sel)
            prompt_feat = F.interpolate(prompt_feat, size=x.shape[2:], mode="bilinear", align_corners=False)

            x_sel = x[idx]
            fused_img, _ = self.fusion(x_sel, prompt_feat)
            fused_img = self.vit2(fused_img)
            x = x.clone()
            x[idx] = fused_img

        fused_2048 = self.proj_back(x)

        logits = self.decoder([skip0, x1, x2, x3, x4, fused_2048])
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(logits, size=image.shape[2:], mode="bilinear", align_corners=False)
        return logits
