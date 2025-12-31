import os
from glob import glob
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform


class PRPDataset(torch.utils.data.Dataset):
    """
    Dataset for Interactive Retinal Laser Photocoagulation area segmentation.

    Each case directory can contain multiple target sub-region masks (gt_*.png).
    Every target mask is treated as an individual sample while sharing the case
    image, and a dynamic click heatmap is generated from an eroded version of
    that mask to avoid overfitting to fixed coordinates.
    """

    def __init__(
        self,
        root_dir: str,
        image_extensions: Optional[List[str]] = None,
        augment: bool = True,
        target_size: Tuple[int, int] = (1280, 1280),
        no_click_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.augment = augment
        self.image_extensions = image_extensions or [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        self.target_size = target_size
        self.no_click_prob = no_click_prob

        self.cases = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
        if not self.cases:
            raise ValueError(f"No case folders found in {root_dir}")

        self.samples: List[Tuple[str, str]] = []
        for case_dir in self.cases:
            mask_paths = sorted(glob(os.path.join(case_dir, "gt_*.png")))
            if not mask_paths:
                raise ValueError(f"No gt_*.png masks found in {case_dir}")
            for mask_path in mask_paths:
                self.samples.append((case_dir, mask_path))

        self.transform = self._build_transform()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def _build_transform(self) -> A.Compose:
        transforms: list[BasicTransform] = []
        if self.augment:
            transforms.extend(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=10,
                        p=0.7,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=0,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=6, val_shift_limit=6, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
                    A.MotionBlur(blur_limit=5, p=0.1),
                    A.RandomGamma(gamma_limit=(90, 110), p=0.2),
                ]
            )

        transforms.append(
            A.Resize(
                height=self.target_size[0],
                width=self.target_size[1],
                interpolation=cv2.INTER_LINEAR,
            )
        )

        return A.Compose(transforms, additional_targets={"mask": "mask"})

    def _load_image(self, case_dir: str) -> np.ndarray:
        image_path = os.path.join(case_dir, "image.png")
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                return image

        # Fallback: try other known extensions but strictly prefer files named "image.*"
        for ext in self.image_extensions:
            candidate = os.path.join(case_dir, f"image{ext}")
            if os.path.exists(candidate):
                image = cv2.imread(candidate)
                if image is not None:
                    return image

        raise FileNotFoundError(
            f"No image found in {case_dir}. Expected image.png or image with extensions {self.image_extensions}"
        )

    def _load_mask(self, mask_path: str) -> np.ndarray:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        return mask_bin.astype(np.uint8)

    @staticmethod
    def _generate_heatmap(height: int, width: int, center: Tuple[int, int], sigma: float = 15.0) -> np.ndarray:
        y = np.arange(0, height, 1, float)
        x = np.arange(0, width, 1, float)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        heatmap = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma ** 2))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.astype(np.float32)

    def _sample_click(self, mask: np.ndarray) -> Tuple[int, int]:
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        if dist.max() <= 0:
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                h, w = mask.shape
                return h // 2, w // 2
            idx = np.random.choice(len(ys))
            return int(ys[idx]), int(xs[idx])

        y0, x0 = np.unravel_index(np.argmax(dist), dist.shape)
        p_periphery = 0.3
        band = 10.0
        eps = 1e-3

        periphery_mask = (dist > 0) & (dist <= band)
        inner_mask = dist > band

        if np.random.rand() < p_periphery and periphery_mask.any():
            ys, xs = np.where(periphery_mask)
            weights = 1.0 / (dist[ys, xs] + eps)
            probs = weights / weights.sum()
            idx = np.random.choice(len(ys), p=probs)
            return int(ys[idx]), int(xs[idx])

        candidate_mask = inner_mask if inner_mask.any() else (dist > 0)
        ys, xs = np.where(candidate_mask)
        if len(ys) == 0:
            h, w = mask.shape
            return h // 2, w // 2

        d = np.sqrt((ys - y0) ** 2 + (xs - x0) ** 2)
        weights = 1.0 / (d + eps)
        probs = weights / weights.sum()
        idx = np.random.choice(len(ys), p=probs)
        return int(ys[idx]), int(xs[idx])

    def __getitem__(self, idx: int):
        case_dir, mask_path = self.samples[idx]
        image = self._load_image(case_dir)
        mask = self._load_mask(mask_path)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        h, w = mask.shape
        click_y, click_x = self._sample_click(mask)
        heatmap = self._generate_heatmap(h, w, (click_y, click_x))

        if self.augment and np.random.rand() < self.no_click_prob:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor_transform = A.Compose(
            [
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
            additional_targets={"heatmap": "mask"},
        )
        tensors = tensor_transform(image=image, mask=mask, heatmap=heatmap)
        image_tensor = tensors["image"]
        mask_tensor = tensors["mask"].unsqueeze(0).float()
        heatmap_tensor = tensors["heatmap"].float().unsqueeze(0)

        return image_tensor, heatmap_tensor, mask_tensor
