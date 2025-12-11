import argparse
import math
import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
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


def main():
    infer_size = (1280, 1280)
    save_size = (1240, 1240)

    parser = argparse.ArgumentParser(description="基于点击的交互式分割预测")
    parser.add_argument("--model", required=True, help="模型权重路径")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--output", required=True, help="输出保存目录")
    parser.add_argument("--threshold", type=float, default=0.4, help="分割阈值，默认0.4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model, device)
    resized_bgr, image_tensor = load_image(args.image, target_size=infer_size)

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
        overlay_bgr = draw_hex_circles(resized_bgr, processed_mask)

        click_state["count"] += 1
        save_results(
            args.output,
            click_state["count"],
            prob_map,
            processed_mask,
            overlay_bgr,
            save_size,
        )

        fig_res, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("原图")
        axes[1].imshow(prob_map, cmap="gray")
        axes[1].set_title("网络输出概率")
        axes[2].imshow(processed_mask, cmap="gray")
        axes[2].set_title("后处理分割")
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
