import os
from glob import glob
from typing import Dict, List, Optional, Tuple

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
    image and auxiliary masks, and a dynamic click heatmap is generated from an
    eroded version of that mask to avoid overfitting to fixed coordinates.

    Additional semantic context is provided by four auxiliary masks named
    aux_0.png ... aux_3.png under each case directory. These masks are stacked
    with the RGB image and Gaussian heatmap to form the 8-channel network input.
    """

    def __init__(
        self,
        root_dir: str,
        image_extensions: Optional[List[str]] = None,
        augment: bool = True,
        target_size: Tuple[int, int] = (1280, 1280),
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.augment = augment
        self.image_extensions = image_extensions or [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        self.target_size = target_size

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
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=8, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                ]
            )

        transforms.append(
            A.Resize(
                height=self.target_size[0],
                width=self.target_size[1],
                interpolation=cv2.INTER_LINEAR,
            )
        )

        additional_targets = {
            "mask": "mask",
            "aux0": "mask",
            "aux1": "mask",
            "aux2": "mask",
            "aux3": "mask",
        }

        return A.Compose(transforms, additional_targets=additional_targets)

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

    def _load_auxiliary_masks(self, case_dir: str) -> List[np.ndarray]:
        aux_masks: List[np.ndarray] = []
        for idx in range(4):
            aux_path = os.path.join(case_dir, f"aux_{idx}.png")
            if not os.path.exists(aux_path):
                raise FileNotFoundError(f"Missing auxiliary mask: {aux_path}")

            aux = cv2.imread(aux_path, cv2.IMREAD_GRAYSCALE)
            if aux is None:
                raise FileNotFoundError(f"Could not read auxiliary mask: {aux_path}")
            _, aux_bin = cv2.threshold(aux, 127, 1, cv2.THRESH_BINARY)
            aux_masks.append(aux_bin.astype(np.uint8))

        return aux_masks

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

        # Increase the selection probability for pixels closer to the mask centroid
        centroid_y = float(ys.mean())
        centroid_x = float(xs.mean())
        distances = np.sqrt((ys - centroid_y) ** 2 + (xs - centroid_x) ** 2)
        # Avoid division by zero; closer pixels get higher weights
        weights = 1.0 / (distances + 1e-3)
        probs = weights / weights.sum()
        idx = np.random.choice(len(ys), p=probs)
        return int(ys[idx]), int(xs[idx])

    def __getitem__(self, idx: int):
        case_dir, mask_path = self.samples[idx]
        image = self._load_image(case_dir)
        mask = self._load_mask(mask_path)
        aux_masks = self._load_auxiliary_masks(case_dir)

        augmented = self.transform(
            image=image,
            mask=mask,
            aux0=aux_masks[0],
            aux1=aux_masks[1],
            aux2=aux_masks[2],
            aux3=aux_masks[3],
        )
        image = augmented["image"]
        mask = augmented["mask"]
        aux_masks = [augmented["aux0"], augmented["aux1"], augmented["aux2"], augmented["aux3"]]

        h, w = mask.shape
        click_y, click_x = self._sample_click(mask)
        heatmap = self._generate_heatmap(h, w, (click_y, click_x))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        additional_targets: Dict[str, str] = {
            "heatmap": "mask",
            "aux0": "mask",
            "aux1": "mask",
            "aux2": "mask",
            "aux3": "mask",
        }
        tensor_transform = A.Compose([ToTensorV2()], additional_targets=additional_targets)
        tensors = tensor_transform(
            image=image,
            mask=mask,
            heatmap=heatmap,
            aux0=aux_masks[0],
            aux1=aux_masks[1],
            aux2=aux_masks[2],
            aux3=aux_masks[3],
        )

        image_tensor = tensors["image"].float()
        mask_tensor = tensors["mask"].unsqueeze(0).float()
        heatmap_tensor = tensors["heatmap"].float().unsqueeze(0)
        aux_tensors = torch.stack(
            [tensors["aux0"], tensors["aux1"], tensors["aux2"], tensors["aux3"]], dim=0
        ).float()

        return image_tensor, heatmap_tensor, aux_tensors, mask_tensor
