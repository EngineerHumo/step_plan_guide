"""
两阶段自动分割与交互演示脚本。

1. 第一阶段：加载 ONNX 语义分割模型，对输入图像执行分割，经过与
   ``prediction_postprecess_firststep.py`` 相同的后处理后，提取绿色区域
   的质心作为初始点击点，并保留最大的红色连通域掩码。
2. 第二阶段：加载基于点击的分割模型（来自 ``model_origin.py``），将
   初始点击点与原图一起缩放到 (1280, 1280) 输入模型，迭代三次获得初步
   分割结果；然后依据第二阶段分割区域，在第一阶段红色区域周围做环形
   光斑排布。
3. 交互：保留原图展示，监听用户点击，可多次生成新的分割与光斑方案，
   文件名自动区分避免覆盖。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from predict import (
    draw_ring_circles,
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
    connected_components,
    process_label,
)


FIRST_STAGE_INPUT = (1024, 1024)
SECOND_STAGE_INPUT = (1280, 1280)


def _centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    y_mean, x_mean = coords.mean(axis=0)
    return int(round(x_mean)), int(round(y_mean))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return mask
    num, comp = connected_components(mask > 0)
    if num <= 1:
        return np.zeros_like(mask)
    areas = [np.sum(comp == idx) for idx in range(1, num)]
    largest_idx = int(np.argmax(areas)) + 1
    return (comp == largest_idx).astype(np.uint8)


def _map_point(point: Tuple[int, int], src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> Tuple[int, int]:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    x, y = point
    mapped_x = int(round(x * dst_w / src_w))
    mapped_y = int(round(y * dst_h / src_h))
    return mapped_x, mapped_y


def _first_stage_inference(
    image_path: Path, sessions, model_input_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]], np.ndarray]:
    rgb_image = _load_image(image_path)
    model_input, pads, processed_shape = _prepare_model_input(rgb_image, model_input_shape)
    labels = _infer_labels(sessions, model_input, pads, processed_shape)

    if processed_shape != rgb_image.shape[:2]:
        labels = cv2.resize(labels.astype(np.uint8), (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    processed_labels = process_label(labels)
    green_center = _centroid(processed_labels == 1)
    largest_red = _largest_component(processed_labels == 3)
    return rgb_image, processed_labels, green_center, largest_red


def _iterative_second_stage(
    model: torch.nn.Module,
    device: torch.device,
    image_tensor: torch.Tensor,
    start_point: Tuple[int, int],
    iterations: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    click_point = start_point
    final_mask: Optional[np.ndarray] = None
    prob_map: Optional[np.ndarray] = None
    empty_mask = np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.uint8)
    for _ in range(iterations):
        prob_map, binary_mask = run_inference(model, device, image_tensor, click_point, threshold)
        processed_mask = postprocess_mask(binary_mask, (click_point[1], click_point[0]))
        final_mask = processed_mask
        new_center = _centroid(processed_mask)
        if new_center is None:
            break
        click_point = new_center
    return (
        prob_map if prob_map is not None else np.array([]),
        final_mask if final_mask is not None else empty_mask,
        _centroid(final_mask if final_mask is not None else empty_mask),
    )


def _save_prediction(
    output_dir: Path,
    name: str,
    prob_map: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    target_size: Tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_path = output_dir / f"{name}_prob.png"
    mask_path = output_dir / f"{name}_mask.png"
    overlay_path = output_dir / f"{name}_overlay.png"

    prob_resized = cv2.resize(prob_map, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    overlay_resized = cv2.resize(overlay, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(str(prob_path), (prob_resized * 255).astype(np.uint8))
    cv2.imwrite(str(mask_path), (mask_resized * 255).astype(np.uint8))
    cv2.imwrite(str(overlay_path), overlay_resized)
    print(f"保存概率图：{prob_path}")
    print(f"保存分割掩码：{mask_path}")
    print(f"保存光斑排布：{overlay_path}")


def _visualize_results(
    bgr_image: np.ndarray,
    prob_map: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原图")
    axes[1].imshow(prob_map, cmap="gray")
    axes[1].set_title("概率图")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("二阶段分割")
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title("环形光斑排布")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="两阶段自动分割与交互演示")
    parser.add_argument("--onnx_dir", type=Path, default=r"C:\work space\prp\predict_demo_260105\fold_1_checkpoint_best.onnx", help="第一阶段 ONNX 模型目录或文件")
    parser.add_argument("--click_model", type=Path, default=r"C:\work space\prp\predict_demo_260105\prp_segmenter.pth", help="第二阶段点击分割模型权重路径")
    parser.add_argument("--image", type=Path, default=r"C:\work space\prp\predict_demo_260105\val\case_0081\image.png", help="输入原始图像路径")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="结果保存目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="分割阈值")
    parser.add_argument("--iterations", type=int, default=3, help="自动迭代次数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备选择")
    args = parser.parse_args()

    first_stage_sessions = _load_sessions(args.onnx_dir)
    device = torch.device(args.device)
    click_model = load_model(str(args.click_model), device)

    rgb_image, _processed_labels, green_center, red_mask = _first_stage_inference(args.image, first_stage_sessions, FIRST_STAGE_INPUT)
    original_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if green_center is None:
        print("警告：未找到绿色区域，自动点击将使用红色区域质心。")
        fallback_center = _centroid(red_mask)
        if fallback_center is None:
            raise RuntimeError("无法确定初始点击点，请确认第一阶段分割结果。")
        green_center = fallback_center

    resized_bgr, image_tensor = load_image(str(args.image), target_size=SECOND_STAGE_INPUT)
    mapped_click = _map_point(green_center, (rgb_image.shape[0], rgb_image.shape[1]), (SECOND_STAGE_INPUT[1], SECOND_STAGE_INPUT[0]))
    red_mask_resized = cv2.resize(red_mask.astype(np.uint8), SECOND_STAGE_INPUT, interpolation=cv2.INTER_NEAREST)

    prob_map, final_mask, final_center = _iterative_second_stage(
        click_model, device, image_tensor, mapped_click, args.iterations, args.threshold
    )
    overlay = draw_ring_circles(resized_bgr, final_mask, reference_mask=red_mask_resized)
    _save_prediction(args.output, "auto_iter", prob_map, final_mask, overlay, (rgb_image.shape[0], rgb_image.shape[1]))
    _visualize_results(resized_bgr, prob_map, final_mask, overlay, "自动分割结果")

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("点击原图以生成新结果")
    plt.axis("off")

    click_state: Dict[str, int] = {"count": 0}

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        print(f"收到用户点击：({x}, {y})，启动分割……")
        mapped = _map_point((x, y), (rgb_image.shape[0], rgb_image.shape[1]), (SECOND_STAGE_INPUT[1], SECOND_STAGE_INPUT[0]))
        prob, mask, center = _iterative_second_stage(
            click_model, device, image_tensor, mapped, args.iterations, args.threshold
        )
        overlay_local = draw_ring_circles(resized_bgr, mask, reference_mask=red_mask_resized)
        click_state["count"] += 1
        name = f"click_{click_state['count']:03d}"
        _save_prediction(args.output, name, prob, mask, overlay_local, (rgb_image.shape[0], rgb_image.shape[1]))
        _visualize_results(resized_bgr, prob, mask, overlay_local, f"用户点击结果 {click_state['count']}")

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    print("窗口已准备好，可多次点击原图生成新的分割与光斑排布。关闭窗口以结束。")
    plt.show()
    fig.canvas.mpl_disconnect(cid)


if __name__ == "__main__":
    main()
