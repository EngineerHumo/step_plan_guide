import os
import random
from glob import glob
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


class PRPDataset(torch.utils.data.Dataset):
    """
    Dataset for Interactive Retinal Laser Photocoagulation area segmentation.

    Each case directory can contain multiple target sub-region masks (gt_*.png).
    A random target mask is selected every epoch, and a dynamic click heatmap is
    generated from an eroded version of that mask to avoid overfitting to fixed
    coordinates.
    """

    def __init__(
        self,
        root_dir: str,
        image_extensions: Optional[List[str]] = None,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.augment = augment
        self.image_extensions = image_extensions or [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        self.cases = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
        if not self.cases:
            raise ValueError(f"No case folders found in {root_dir}")

        self.transform = self._build_transform()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.cases)

    def _build_transform(self) -> A.Compose:
        transforms = [
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.7, border_mode=cv2.BORDER_REFLECT),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=8, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ]
        return A.Compose(transforms, additional_targets={"mask": "mask"})

    def _load_image(self, case_dir: str) -> np.ndarray:
        for ext in self.image_extensions:
            candidates = [p for p in glob(os.path.join(case_dir, f"*{ext}")) if "gt_" not in os.path.basename(p)]
            if candidates:
                return cv2.imread(candidates[0])
        raise FileNotFoundError(f"No image found in {case_dir} with extensions {self.image_extensions}")

    def _load_random_mask(self, case_dir: str) -> np.ndarray:
        mask_paths = sorted(glob(os.path.join(case_dir, "gt_*.png")))
        if not mask_paths:
            raise FileNotFoundError(f"No gt_*.png masks found in {case_dir}")
        mask_path = random.choice(mask_paths)
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
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        ys, xs = np.where(eroded > 0)
        if len(ys) == 0:
            ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            h, w = mask.shape
            return h // 2, w // 2
        idx = random.randint(0, len(ys) - 1)
        return int(ys[idx]), int(xs[idx])

    def __getitem__(self, idx: int):
        case_dir = self.cases[idx]
        image = self._load_image(case_dir)
        mask = self._load_random_mask(case_dir)

        if self.augment:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        h, w = mask.shape
        click_y, click_x = self._sample_click(mask)
        heatmap = self._generate_heatmap(h, w, (click_y, click_x))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        tensor_transform = A.Compose([ToTensorV2()])
        tensors = tensor_transform(image=image, mask=mask, heatmap=heatmap)
        image_tensor = tensors["image"]
        mask_tensor = tensors["mask"].unsqueeze(0).float()
        #heatmap_tensor = tensors["heatmap"].unsqueeze(0)
        heatmap_tensor = torch.from_numpy(tensors["heatmap"]).float().unsqueeze(0)

        return image_tensor, heatmap_tensor, mask_tensor
