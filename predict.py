import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from model_origin import PRPSegmenter
from prediction_postprecess_firststep import (
    _infer_labels,
    _load_image,
    _load_sessions,
    _prepare_model_input,
    connected_components,
    process_label,
)


def load_model(model_path: str, device: torch.device) -> PRPSegmenter:
    model = PRPSegmenter(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_image(image_path: str, target_size: Tuple[int, int]) -> Tuple[np.ndarray, torch.Tensor]:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0)
    return bgr, tensor


def generate_heatmap(height: int, width: int, center: Tuple[int, int], sigma: float = 15.0) -> np.ndarray:
    y = np.arange(0, height, 1, float)
    x = np.arange(0, width, 1, float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    heatmap = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma ** 2))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.astype(np.float32)


def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    filtered = np.zeros_like(mask)
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > min_area:
            filtered[labels == label] = 1
    return filtered


def postprocess_mask(
    binary_mask: np.ndarray,
    click_point: Tuple[int, int],
    dilation_size: int = 5,
    min_area: int = 100,
    distance_threshold: int = 100,
) -> np.ndarray:
    mask = binary_mask.astype(np.uint8)
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = _filter_small_components(mask, min_area=min_area)
    if mask.max() == 0:
        return mask

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    click_y, click_x = click_point
    click_label = labels[click_y, click_x] if 0 <= click_y < labels.shape[0] and 0 <= click_x < labels.shape[1] else 0

    kept_labels = {largest_label}
    if click_label > 0:
        kept_labels.add(click_label)

    kept_mask = np.isin(labels, list(kept_labels)).astype(np.uint8)

    while True:
        dist_map = cv2.distanceTransform(1 - kept_mask, cv2.DIST_L2, 5)
        new_labels: List[int] = []
        for label in range(1, num):
            if label in kept_labels:
                continue
            component_dist = dist_map[labels == label].min()
            if component_dist < distance_threshold:
                new_labels.append(label)
        if not new_labels:
            break
        kept_labels.update(new_labels)
        kept_mask = np.isin(labels, list(kept_labels)).astype(np.uint8)

    return kept_mask


def _centroid_from_mask(mask: np.ndarray) -> Tuple[int, int]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("无法在空掩码上计算质心")
    y_mean, x_mean = coords.mean(axis=0)
    return int(round(x_mean)), int(round(y_mean))


def generate_ring_centers(
    seg_mask: np.ndarray,
    reference_mask: Optional[np.ndarray] = None,
    diameter: int = 15,
) -> List[Tuple[int, int]]:
    if seg_mask.max() == 0:
        return []

    spacing = 2 * diameter
    try:
        origin = _centroid_from_mask(reference_mask if reference_mask is not None else seg_mask)
    except ValueError:
        return []

    h, w = seg_mask.shape
    origin_x, origin_y = origin
    if not (0 <= origin_x < w and 0 <= origin_y < h) or seg_mask[origin_y, origin_x] == 0:
        dist, labels = cv2.distanceTransformWithLabels(
            seg_mask.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        if dist.max() == 0:
            return []
        nearest_label = labels[origin_y, origin_x] - 1 if 0 <= origin_y < h and 0 <= origin_x < w else labels.max() - 1
        coords = np.column_stack(np.where(seg_mask > 0))
        coords = coords[nearest_label] if 0 <= nearest_label < len(coords) else coords[0]
        origin_y, origin_x = int(coords[0]), int(coords[1])

    coords = np.column_stack(np.where(seg_mask > 0))
    max_radius = int(np.linalg.norm(coords - np.array([origin_y, origin_x]), axis=1).max())

    centers: List[Tuple[int, int]] = []
    ring = 0
    while True:
        current_r = ring * spacing
        if current_r > max_radius:
            break
        if current_r == 0:
            candidate = (origin_x, origin_y)
            if seg_mask[origin_y, origin_x] > 0:
                centers.append(candidate)
        else:
            circumference = 2 * math.pi * current_r
            num_points = max(1, int(math.floor(circumference / spacing)))
            angle_step = 2 * math.pi / num_points
            for i in range(num_points):
                angle = i * angle_step
                x = int(round(origin_x + current_r * math.cos(angle)))
                y = int(round(origin_y + current_r * math.sin(angle)))
                if not (0 <= x < w and 0 <= y < h):
                    continue
                if seg_mask[y, x] == 0:
                    continue
                if all((x - cx) ** 2 + (y - cy) ** 2 >= spacing ** 2 for cx, cy in centers):
                    centers.append((x, y))
        ring += 1

    return centers


def draw_ring_circles(
    image_bgr: np.ndarray,
    seg_mask: np.ndarray,
    reference_mask: Optional[np.ndarray] = None,
    diameter: int = 15,
) -> np.ndarray:
    centers = generate_ring_centers(seg_mask, reference_mask=reference_mask, diameter=diameter)
    overlay = image_bgr.copy()
    for center in centers:
        cv2.circle(overlay, center, diameter // 2, (255, 0, 0), thickness=2)
    return overlay


def _largest_red_mask(labels: np.ndarray) -> np.ndarray:
    if labels.max() == 0:
        return np.zeros_like(labels, dtype=np.uint8)
    num, comps = connected_components(labels == 3)
    if num <= 1:
        return np.zeros_like(labels, dtype=np.uint8)
    areas = [np.sum(comps == idx) for idx in range(1, num)]
    largest_idx = int(np.argmax(areas)) + 1
    return (comps == largest_idx).astype(np.uint8)


def _first_stage_red_mask(image_path: Path, onnx_dir: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """Run ONNX first stage to obtain the largest red area mask and resize to match second stage input."""

    rgb_image = _load_image(image_path)
    model_input, pads, processed_shape = _prepare_model_input(rgb_image, target_size)
    sessions = _load_sessions(onnx_dir)
    labels = _infer_labels(sessions, model_input, pads, processed_shape)

    if processed_shape != rgb_image.shape[:2]:
        labels = cv2.resize(labels.astype(np.uint8), (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    processed_labels = process_label(labels)
    red_mask = _largest_red_mask(processed_labels)
    resized_red = cv2.resize(red_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
    return resized_red


def run_inference(
    model: PRPSegmenter,
    device: torch.device,
    image_tensor: torch.Tensor,
    click_point: Tuple[int, int],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    _, _, height, width = image_tensor.shape
    heatmap = generate_heatmap(height, width, (click_point[1], click_point[0]))
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(image_tensor.to(device), heatmap_tensor)
    prob_map = preds.squeeze().cpu().numpy()
    binary_mask = (prob_map >= threshold).astype(np.uint8)
    return prob_map, binary_mask


def save_results(
    output_dir: str,
    click_index: int,
    prob_map: np.ndarray,
    processed_mask: np.ndarray,
    overlay_bgr: np.ndarray,
    save_size: Tuple[int, int],
):
    os.makedirs(output_dir, exist_ok=True)
    prob_path = os.path.join(output_dir, f"prediction_prob_{click_index}.png")
    mask_path = os.path.join(output_dir, f"segmentation_mask_{click_index}.png")
    overlay_path = os.path.join(output_dir, f"ring_overlay_{click_index}.png")

    prob_resized = cv2.resize(prob_map, save_size, interpolation=cv2.INTER_LINEAR)
    prob_vis = (prob_resized * 255).astype(np.uint8)

    mask_resized = cv2.resize(processed_mask, save_size, interpolation=cv2.INTER_NEAREST)
    overlay_resized = cv2.resize(overlay_bgr, save_size, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(prob_path, prob_vis)
    cv2.imwrite(mask_path, (mask_resized * 255).astype(np.uint8))
    cv2.imwrite(overlay_path, overlay_resized)
    print(f"保存概率图：{prob_path}")
    print(f"保存分割掩码：{mask_path}")
    print(f"保存环形排布结果：{overlay_path}")


def main():
    infer_size = (1280, 1280)
    save_size = (1240, 1240)

    parser = argparse.ArgumentParser(description="基于点击的交互式分割预测")
    parser.add_argument("--model", default="C:/work space/prp/predict/best_model.pth", help="模型权重路径")
    parser.add_argument("--image", default="C:/work space/prp/predict/val/case_0081/image.png", help="输入图像路径")
    parser.add_argument("--output", default="C:/work space/prp/predict/run", help="输出保存目录")
    parser.add_argument("--onnx_dir", type=str, default=None, help="第一阶段 ONNX 模型目录或文件，用于获取红色区域掩码")
    parser.add_argument("--threshold", type=float, default=0.5, help="分割阈值，默认0.4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model, device)
    resized_bgr, image_tensor = load_image(args.image, target_size=infer_size)

    reference_mask: Optional[np.ndarray] = None
    if args.onnx_dir is not None:
        try:
            reference_mask = _first_stage_red_mask(Path(args.image), Path(args.onnx_dir), infer_size)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"警告：第一阶段推理失败，无法获取红色区域。错误：{exc}")
            reference_mask = None

    height, width = image_tensor.shape[2:]
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("点击图像以生成分割")
    plt.axis("off")

    click_state = {"count": 0}

    def on_click(event):
        if event.inaxes != ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        print(f"收到点击：({x}, {y})，开始推理……")
        prob_map, binary_mask = run_inference(model, device, image_tensor, (x, y), args.threshold)
        processed_mask = postprocess_mask(binary_mask, (y, x))
        overlay_bgr = draw_ring_circles(resized_bgr, processed_mask, reference_mask=reference_mask)

        click_state["count"] += 1
        save_results(
            args.output,
            click_state["count"],
            prob_map,
            processed_mask,
            overlay_bgr,
            save_size,
        )

        fig_res, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("原图")
        axes[1].imshow(prob_map, cmap="gray")
        axes[1].set_title("网络输出概率")
        axes[2].imshow(processed_mask, cmap="gray")
        axes[2].set_title("后处理分割")
        axes[3].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
        axes[3].set_title("环形排布")
        for a in axes:
            a.axis("off")
        plt.tight_layout()
        plt.show()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    print("窗口已准备好，请在图像上点击以开始分割。关闭窗口以结束。")
    plt.show()
    fig.canvas.mpl_disconnect(cid)


if __name__ == "__main__":
    main()
