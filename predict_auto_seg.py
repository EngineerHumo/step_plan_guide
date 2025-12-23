"""
基于两阶段的自动分割与交互式细化脚本。

阶段 1：使用 ONNX 语义分割模型对输入图像进行初步分割并经过与
``prediction_postprecess_firststep.py`` 相同的后处理，自动找到绿色区域
（标签 1）的几何中心，作为模拟点击点。

阶段 2：加载基于点击的交互式分割模型，以上述中心点作为初始点击执
行推理并展示结果；随后监听用户点击，生成新的分割结果与可视化图并
保存，文件名增加标识以避免覆盖。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from predict import (
    draw_hex_circles,
    load_image,
    load_model,
    postprocess_mask,
    run_inference,
)
from prediction_postprecess_firststep import (
    _infer_labels,
    _load_image,
    _load_sessions,
    _prepare_model_input,
    process_label,
)


FIRST_STAGE_INPUT_SHAPE: Tuple[int, int] = (1280, 1280)
SECOND_STAGE_INPUT_SHAPE: Tuple[int, int] = (1280, 1280)
SECOND_STAGE_SAVE_SHAPE: Tuple[int, int] = (1240, 1240)


def compute_green_center(labels: np.ndarray) -> Tuple[int, int]:
    """计算绿色区域（标签 1）的质心。若存在多个连通域，则取面积最大者。"""

    green_mask = labels == 1
    if not np.any(green_mask):
        raise ValueError("预测结果中未找到绿色区域，无法确定初始点击点。")

    num, components = cv2.connectedComponents(green_mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        # 只有背景
        raise ValueError("绿色区域连通域数量为 0，无法确定初始点击点。")

    max_label: Optional[int] = None
    max_area = -1
    for label in range(1, num):
        area = int(np.sum(components == label))
        if area > max_area:
            max_area = area
            max_label = label

    if max_label is None or max_area <= 0:
        raise ValueError("未能识别有效的绿色连通域。")

    target_mask = components == max_label
    moments = cv2.moments(target_mask.astype(np.uint8))
    if moments["m00"] == 0:
        raise ValueError("绿色连通域的矩计算失败。")

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def first_stage_inference(
    image_path: Path, onnx_dir: Path, input_shape: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """执行第一阶段语义分割并返回后处理标签与绿色区域中心。"""

    image_rgb = _load_image(image_path)
    model_input, pads, processed_shape = _prepare_model_input(image_rgb, input_shape)
    sessions = _load_sessions(onnx_dir)
    labels = _infer_labels(sessions, model_input, pads, processed_shape)

    if processed_shape != image_rgb.shape[:2]:
        labels = cv2.resize(
            labels.astype(np.uint8),
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    processed_labels = process_label(labels)
    center = compute_green_center(processed_labels)
    return processed_labels, center


def save_results(
    output_dir: Path,
    click_id: str,
    prob_map: np.ndarray,
    processed_mask: np.ndarray,
    overlay_bgr: np.ndarray,
    save_size: Tuple[int, int],
) -> None:
    """保存概率图、掩码和六边形覆盖，可接受字符串 ID 以避免覆盖。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    prob_path = output_dir / f"prediction_prob_{click_id}.png"
    mask_path = output_dir / f"segmentation_mask_{click_id}.png"
    overlay_path = output_dir / f"hex_overlay_{click_id}.png"

    prob_resized = cv2.resize(prob_map, save_size, interpolation=cv2.INTER_LINEAR)
    prob_vis = (prob_resized * 255).astype(np.uint8)

    mask_resized = cv2.resize(processed_mask, save_size, interpolation=cv2.INTER_NEAREST)
    overlay_resized = cv2.resize(overlay_bgr, save_size, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(str(prob_path), prob_vis)
    cv2.imwrite(str(mask_path), (mask_resized * 255).astype(np.uint8))
    cv2.imwrite(str(overlay_path), overlay_resized)

    print(f"保存概率图：{prob_path}")
    print(f"保存分割掩码：{mask_path}")
    print(f"保存六边形填充结果：{overlay_path}")


def generate_and_show(
    model: torch.nn.Module,
    device: torch.device,
    resized_bgr: np.ndarray,
    image_tensor: torch.Tensor,
    click_point: Tuple[int, int],
    threshold: float,
    output_dir: Path,
    click_id: str,
    save_size: Tuple[int, int],
) -> None:
    """执行点击推理、保存结果并弹出可视化窗口。"""

    prob_map, binary_mask = run_inference(model, device, image_tensor, click_point, threshold)
    processed_mask = postprocess_mask(binary_mask, (click_point[1], click_point[0]))
    overlay_bgr = draw_hex_circles(resized_bgr, processed_mask)

    save_results(output_dir, click_id, prob_map, processed_mask, overlay_bgr, save_size)

    fig_res, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原图")
    axes[1].imshow(prob_map, cmap="gray")
    axes[1].set_title("网络输出概率")
    axes[2].imshow(processed_mask, cmap="gray")
    axes[2].set_title("后处理分割")
    axes[3].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    axes[3].set_title("六边形填充")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="两阶段自动+交互式分割")
    parser.add_argument("--onnx_dir", type=Path, required=True, help="ONNX 模型文件或目录")
    parser.add_argument("--click_model", type=Path, required=True, help="基于点击的分割模型权重")
    parser.add_argument("--image", type=Path, required=True, help="输入图像路径")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="分割阈值，默认 0.5")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    processed_labels, auto_center = first_stage_inference(
        args.image, args.onnx_dir, FIRST_STAGE_INPUT_SHAPE
    )
    print(f"自动识别的初始点击点（x, y）：{auto_center}")

    resized_bgr, image_tensor = load_image(str(args.image), target_size=SECOND_STAGE_INPUT_SHAPE)
    model = load_model(str(args.click_model), device)

    generate_and_show(
        model,
        device,
        resized_bgr,
        image_tensor,
        auto_center,
        args.threshold,
        args.output,
        "auto_center",
        SECOND_STAGE_SAVE_SHAPE,
    )

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("点击图像以生成新的分割")
    plt.axis("off")

    click_state = {"count": 0}

    def on_click(event):
        if event.inaxes != ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        click_state["count"] += 1
        click_id = f"click_{click_state['count']}"
        print(f"收到点击：({x}, {y})，生成 {click_id} 的分割结果……")
        generate_and_show(
            model,
            device,
            resized_bgr,
            image_tensor,
            (x, y),
            args.threshold,
            args.output,
            click_id,
            SECOND_STAGE_SAVE_SHAPE,
        )

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    print("窗口已准备好，关闭窗口结束交互。")
    plt.show()
    fig.canvas.mpl_disconnect(cid)


if __name__ == "__main__":
    main()
