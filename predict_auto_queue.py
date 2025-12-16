import argparse
import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from model import PRPSegmenter


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


def generate_hex_centers(mask: np.ndarray, radius: int = 10, min_gap: int = 10) -> List[Tuple[int, int]]:
    if mask.max() == 0:
        return []
    effective_spacing = 2 * radius + min_gap
    vertical_spacing = effective_spacing * math.sqrt(3) / 2

    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    centers: List[Tuple[int, int]] = []

    row = 0
    y = y_min + radius
    while y <= y_max - radius:
        offset = 0.0 if row % 2 == 0 else effective_spacing / 2
        x = x_min + radius + offset
        while x <= x_max - radius:
            yi = int(round(y))
            xi = int(round(x))
            if dist_map[yi, xi] >= radius:
                centers.append((xi, yi))
            x += effective_spacing
        row += 1
        y = y_min + radius + row * vertical_spacing

    return centers


def draw_hex_circles(image_bgr: np.ndarray, mask: np.ndarray, radius: int = 10, min_gap: int = 10) -> np.ndarray:
    centers = generate_hex_centers(mask, radius=radius, min_gap=min_gap)
    overlay = image_bgr.copy()
    for center in centers:
        cv2.circle(overlay, center, radius, (255, 0, 0), thickness=2)
    return overlay


def calculate_mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    y_mean, x_mean = coords.mean(axis=0)
    return int(round(x_mean)), int(round(y_mean))


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
    overlay_path = os.path.join(output_dir, f"hex_overlay_{click_index}.png")

    prob_resized = cv2.resize(prob_map, save_size, interpolation=cv2.INTER_LINEAR)
    prob_vis = (prob_resized * 255).astype(np.uint8)

    mask_resized = cv2.resize(processed_mask, save_size, interpolation=cv2.INTER_NEAREST)
    overlay_resized = cv2.resize(overlay_bgr, save_size, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(prob_path, prob_vis)
    cv2.imwrite(mask_path, (mask_resized * 255).astype(np.uint8))
    cv2.imwrite(overlay_path, overlay_resized)
    print(f"保存概率图：{prob_path}")
    print(f"保存分割掩码：{mask_path}")
    print(f"保存六边形填充结果：{overlay_path}")


def save_merge_overlay(
    output_dir: str, overlays: List[np.ndarray], save_size: Tuple[int, int], base_name: str
):
    if not overlays:
        return

    width, height = save_size
    canvas = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
    for idx, overlay in enumerate(overlays[:9]):
        resized = cv2.resize(overlay, save_size, interpolation=cv2.INTER_LINEAR)
        row, col = divmod(idx, 3)
        y0, y1 = row * height, (row + 1) * height
        x0, x1 = col * width, (col + 1) * width
        canvas[y0:y1, x0:x1] = resized

    output_path = os.path.join(output_dir, f"{base_name}_merge.png")
    cv2.imwrite(output_path, canvas)
    print(f"保存合并画布：{output_path}")


def main():
    infer_size = (1280, 1280)
    save_size = (1240, 1240)

    parser = argparse.ArgumentParser(description="基于点击的交互式分割预测")
    parser.add_argument("--model", default="/home/wensheng/jiaqi/step_plan_guide/output_test5_epoch200_cnn_vit/best_model.pth", help="模型权重路径")
    parser.add_argument("--image", default="/home/wensheng/jiaqi/step_plan_guide/autopredict_demodata", help="输入图像路径")
    parser.add_argument("--output", default=None, help="输出保存目录（若为空则保存在原图路径下）")
    parser.add_argument("--threshold", type=float, default=0.5, help="分割阈值，默认0.4")
    parser.add_argument("--iterations", type=int, default=6, help="每个点的迭代次数")
    parser.add_argument("--circle_diameter", type=int, default=20, help="填充圆形的直径")
    parser.add_argument("--circle_gap", type=int, default=15, help="填充圆形之间的最小距离")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model, device)

    input_path = args.image
    if os.path.isdir(input_path):
        image_files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
    elif os.path.isfile(input_path):
        image_files = [input_path]
    else:
        raise FileNotFoundError(f"未找到输入路径：{input_path}")

    for image_path in image_files:
        resized_bgr, image_tensor = load_image(image_path, target_size=infer_size)

        initial_points = [
            (320, 320),
            (320, 640),
            (320, 960),
            (640, 320),
            (640, 640),
            (640, 960),
            (960, 320),
            (960, 640),
            (960, 960),
        ]

        overlays = []
        radius = max(1, args.circle_diameter // 2)
        for idx, point in enumerate(initial_points, start=1):
            current_click = point
            prob_map = None
            processed_mask = None
            for _ in range(args.iterations):
                prob_map, binary_mask = run_inference(
                    model, device, image_tensor, current_click, args.threshold
                )
                processed_mask = postprocess_mask(
                    binary_mask, (current_click[1], current_click[0])
                )
                centroid = calculate_mask_centroid(processed_mask)
                if centroid is None:
                    break
                current_click = centroid

            if prob_map is None or processed_mask is None:
                continue

            overlay_bgr = draw_hex_circles(
                resized_bgr, processed_mask, radius=radius, min_gap=args.circle_gap
            )
            overlays.append((overlay_bgr, prob_map, processed_mask))

        base_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = (
            args.output if args.output else os.path.join(base_dir, base_name)
        )

        for idx, (overlay_img, prob_map, processed_mask) in enumerate(
            overlays, start=1
        ):
            save_results(
                output_dir,
                idx,
                prob_map,
                processed_mask,
                overlay_img,
                save_size,
            )

        merge_overlays = [item[0] for item in overlays]
        save_merge_overlay(output_dir, merge_overlays, save_size, base_name)


if __name__ == "__main__":
    main()
