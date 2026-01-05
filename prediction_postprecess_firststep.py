'''
加载onnx，对模型进行初步语义分割，找到启示点击点。
运行方式示例::

    python prediction_postprecess.py \
        --onnx_dir /path/to/onnx_models \
        --images_dir /path/to/images\ 
        --output_dir /path/to/output
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort


# --------------------------- 基础配色与常量定义 ---------------------------
# 颜色到标签的映射，使用 BGR 顺序以匹配 OpenCV 的默认读取方式。
COLOR_TO_LABEL = {
    (0, 0, 0): 0,  # 背景（黑）
    (0, 255, 0): 1,  # 绿色
    (0, 255, 255): 2,  # 黄色（BGR）
    (0, 0, 255): 3,  # 红色
}
LABEL_PRIORITY = [0, 1, 2, 3]
LABEL_TO_COLOR = {
    0: np.array((0, 0, 0), dtype=np.uint8),
    1: np.array((0, 255, 0), dtype=np.uint8),
    2: np.array((0, 255, 255), dtype=np.uint8),
    3: np.array((0, 0, 255), dtype=np.uint8),
}
TARGET_SIZE: Tuple[int, int] = (1240, 1240)


# --------------------------- 通用工具函数 ---------------------------

def _load_image(path: Path) -> np.ndarray:
    """读取原始 RGB 图像。"""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _pad_to_shape(image: np.ndarray, target_shape: Sequence[int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """将图像填充到目标尺寸，返回填充后的图像与对应的边界。"""

    target_h, target_w = target_shape
    h, w = image.shape[:2]
    if h > target_h or w > target_w:
        raise ValueError(
            f"图像尺寸 {h}x{w} 超过模型期望尺寸 {target_h}x{target_w}，无法直接填充。"
        )

    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def _prepare_model_input(
    image: np.ndarray, target_shape: Sequence[int]
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int]]:
    """对图像执行尺寸调整、填充与归一化以匹配 ONNX 模型输入。

    当原图尺寸超过模型输入尺寸时，先按双线性插值缩放到目标尺寸；当
    原图较小或与目标尺寸一致时，保持原尺寸并通过 `_pad_to_shape` 做边
    缘填充。返回模型输入张量、填充信息以及参与推理的图像尺寸。
    """

    target_h, target_w = target_shape
    height, width = image.shape[:2]

    if height > target_h or width > target_w:
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    processed_shape = resized.shape[:2]
    padded, pads = _pad_to_shape(resized, target_shape)
    input_array = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(input_array, axis=0), pads, processed_shape


def _load_sessions(onnx_dir: Path) -> List[ort.InferenceSession]:
    """加载目录中的所有 ONNX 模型。"""

    if onnx_dir.is_file():
        onnx_files = [onnx_dir]
    else:
        onnx_files = sorted(p for p in onnx_dir.glob("*.onnx"))

    if not onnx_files:
        raise FileNotFoundError(f"在 {onnx_dir} 未找到任何 .onnx 模型")

    sessions: List[ort.InferenceSession] = []
    for path in onnx_files:
        sessions.append(ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]))
    return sessions


def _infer_labels(
    sessions: Sequence[ort.InferenceSession],
    model_input: np.ndarray,
    pads: Tuple[int, int, int, int],
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """执行模型推理并转换为标签图。"""

    accumulated = None
    input_name_cache = [session.get_inputs()[0].name for session in sessions]
    for session, input_name in zip(sessions, input_name_cache):
        ort_inputs = {input_name: model_input}
        output = session.run(None, ort_inputs)[0]
        if output.ndim != 4:
            raise ValueError("ONNX 模型输出维度不符，期望形状为 (N, C, H, W)")
        prediction = output[0]
        accumulated = prediction if accumulated is None else accumulated + prediction

    mean_prediction = accumulated / len(sessions)

    pad_top, pad_bottom, pad_left, pad_right = pads
    _, padded_h, padded_w = mean_prediction.shape
    h, w = output_shape

    start_h = pad_top
    end_h = padded_h - pad_bottom
    start_w = pad_left
    end_w = padded_w - pad_right

    cropped = mean_prediction[:, start_h:end_h, start_w:end_w]
    if cropped.shape[1:] != (h, w):
        raise ValueError("裁剪后的尺寸与原始图像不匹配")

    channel_indices = np.argmax(cropped, axis=0).astype(np.uint8)
    channel_to_label = np.array([0, 3, 2, 1], dtype=np.uint8)
    labels = channel_to_label[channel_indices]
    return labels


def save_label_image(path: Path, labels: np.ndarray) -> None:
    """按照标签优先级将标签重新映射为彩色 PNG 并保存。"""

    h, w = labels.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    for label in LABEL_PRIORITY:
        mask = labels == label
        if not np.any(mask):
            continue
        output[mask] = LABEL_TO_COLOR[label]
    resized = cv2.resize(output, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), resized)


def connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """对二值掩膜执行 8 邻域连通域分割。"""

    num, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return num, comp


def boundary_band(labels: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """利用形态学梯度计算所有类别边界组成的窄带。"""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    band = np.zeros_like(labels, dtype=bool)
    for value in range(4):
        mask = (labels == value).astype(np.uint8)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        band |= dilated != eroded
    return band


def majority_filter_on_band(labels: np.ndarray, band: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """在边界窄带上执行多数平滑。"""

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    counts = []
    for value in range(4):
        mask = (labels == value).astype(np.float32)
        count = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        counts.append(count)
    stacked = np.stack(counts, axis=-1)
    modes = np.argmax(stacked, axis=-1).astype(np.uint8)
    smoothed = labels.copy()
    smoothed[band] = modes[band]
    return smoothed


def dilate_red(labels: np.ndarray) -> np.ndarray:
    """对红色区域执行 1 像素膨胀。"""

    red_mask = labels == 3
    if not np.any(red_mask):
        return labels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(red_mask.astype(np.uint8), kernel)
    result = labels.copy()
    result[dilated.astype(bool)] = 3
    return result


def replace_component_with_neighbors(labels: np.ndarray, component_mask: np.ndarray, value: int) -> None:
    """将指定连通域替换为其相邻像素中最常见的颜色。"""

    if not np.any(component_mask):
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    border = cv2.dilate(component_mask.astype(np.uint8), kernel).astype(bool)
    border &= ~component_mask
    if not np.any(border):
        labels[component_mask] = 0
        return
    neighbors = labels[border]
    counts = np.bincount(neighbors, minlength=4)
    counts[value] = 0
    new_value = int(np.argmax(counts))
    labels[component_mask] = new_value


def clean_green_components(labels: np.ndarray) -> None:
    """删除面积不足的绿色连通域，并依邻域颜色填补。"""

    green_mask = labels == 1
    total = int(np.sum(green_mask))
    if total == 0:
        return
    threshold = max(int(total * 0.1), 1)
    num, comp = connected_components(green_mask)
    for idx in range(1, num):
        component_mask = comp == idx
        if int(np.sum(component_mask)) < threshold:
            replace_component_with_neighbors(labels, component_mask, 1)


def keep_largest_component(labels: np.ndarray, value: int) -> None:
    """保留指定颜色的最大连通域，其余区域依邻域颜色重赋值。"""

    mask = labels == value
    if not np.any(mask):
        return
    num, comp = connected_components(mask)
    areas = [np.sum(comp == idx) for idx in range(1, num)]
    if not areas:
        return
    largest_idx = int(np.argmax(areas)) + 1
    for idx in range(1, num):
        if idx == largest_idx:
            continue
        component_mask = comp == idx
        replace_component_with_neighbors(labels, component_mask, value)


def fill_removed_regions(labels: np.ndarray, removed_mask: np.ndarray, target_value: int) -> None:
    """将形态学开运算移除的像素填充为最近的其他颜色。"""

    if not np.any(removed_mask):
        return
    binary = (labels == target_value).astype(np.uint8)
    if np.all(binary == 1):
        labels[removed_mask] = target_value
        return
    dist, indices = cv2.distanceTransformWithLabels(
        binary, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    zero_coords = np.column_stack(np.where(binary == 0))
    target_indices = indices[removed_mask] - 1
    target_indices = np.clip(target_indices, 0, len(zero_coords) - 1)
    nearest_coords = zero_coords[target_indices]
    new_values = labels[nearest_coords[:, 0], nearest_coords[:, 1]]
    labels[removed_mask] = new_values


def opening_and_refill(labels: np.ndarray, value: int, radius: int = 4) -> None:
    """对指定颜色执行开运算并填补空缺。"""

    mask = labels == value
    if not np.any(mask):
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    removed = mask & ~opened
    labels[mask] = value
    fill_removed_regions(labels, removed, value)


def area_preserving_rethreshold(labels: np.ndarray) -> None:
    """在黄绿窄带内执行面积守恒的高斯重阈值，仅调整黄绿边界。"""

    original_black = labels == 0
    original_red = labels == 3
    yellow_mask = labels == 2
    green_mask = labels == 1
    yellow_green = yellow_mask | green_mask
    if not np.any(yellow_green):
        return

    protect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    protected = cv2.dilate((original_black | original_red).astype(np.uint8), protect_kernel).astype(bool)
    movable = yellow_green & ~protected
    if not np.any(movable):
        labels[original_black] = 0
        labels[original_red] = 3
        return

    kernel_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    dilated_y = cv2.dilate(yellow_mask.astype(np.uint8), kernel_band).astype(bool)
    dilated_g = cv2.dilate(green_mask.astype(np.uint8), kernel_band).astype(bool)
    band = movable & dilated_y & dilated_g
    if not np.any(band):
        band = movable

    yellow_fixed = yellow_mask & ~band
    yellow_target = int(np.sum(yellow_mask))
    yellow_to_allocate = yellow_target - int(np.sum(yellow_fixed))
    if yellow_to_allocate <= 0:
        labels[band] = 1
        labels[yellow_fixed] = 2
        labels[original_black] = 0
        labels[original_red] = 3
        return

    band_size = int(np.sum(band))
    if yellow_to_allocate >= band_size:
        labels[band] = 2
        labels[yellow_fixed] = 2
        labels[original_black] = 0
        labels[original_red] = 3
        return

    blur = cv2.GaussianBlur(
        yellow_mask.astype(np.float32),
        (0, 0),
        sigmaX=12.0,
        sigmaY=12.0,
        borderType=cv2.BORDER_REPLICATE,
    )
    values = blur[band]
    flat_band_indices = np.flatnonzero(band)

    partition_index = len(values) - yellow_to_allocate
    threshold_value = np.partition(values, partition_index)[partition_index]
    larger = values > threshold_value
    selected_count = int(np.sum(larger))

    equals = np.where(values == threshold_value)[0]
    need = yellow_to_allocate - selected_count
    if need > 0:
        tie_indices = equals[:need]
    else:
        tie_indices = np.array([], dtype=int)

    new_yellow_mask = np.zeros_like(yellow_mask, dtype=bool)
    if selected_count > 0:
        selected_idx = np.flatnonzero(larger)
        new_yellow_mask.flat[flat_band_indices[selected_idx]] = True
    if need > 0:
        new_yellow_mask.flat[flat_band_indices[tie_indices]] = True

    labels[band] = 1
    labels[new_yellow_mask] = 2
    labels[yellow_fixed] = 2
    labels[original_black] = 0
    labels[original_red] = 3



def clean_yellow_components(labels: np.ndarray) -> None:
    """移除过小的黄色连通域，并依邻域颜色填补。"""

    yellow_mask = labels == 2
    total = int(np.sum(yellow_mask))
    if total == 0:
        return

    threshold = max(int(total * 0.1), 1)
    num, comp = connected_components(yellow_mask)
    if num <= 1:
        return

    for idx in range(1, num):
        component_mask = comp == idx
        if int(np.sum(component_mask)) < threshold:
            replace_component_with_neighbors(labels, component_mask, 2)


def remove_small_components(labels: np.ndarray, min_size: int) -> None:
    """移除所有小于给定面积阈值的连通域，并依邻域颜色重赋值。"""

    for value in range(4):
        mask = labels == value
        if not np.any(mask):
            continue
        num, comp = connected_components(mask)
        if num <= 1:
            continue
        for idx in range(1, num):
            component_mask = comp == idx
            if int(np.sum(component_mask)) < min_size:
                replace_component_with_neighbors(labels, component_mask, value)


def process_label(labels: np.ndarray) -> np.ndarray:
    """对单张标签图执行完整的后处理流程。"""

    band = boundary_band(labels)
    labels = majority_filter_on_band(labels, band)
    labels = dilate_red(labels)
    clean_green_components(labels)
    keep_largest_component(labels, 3)
    keep_largest_component(labels, 0)
    opening_and_refill(labels, 1, radius=4)
    opening_and_refill(labels, 2, radius=4)
    area_preserving_rethreshold(labels)
    clean_yellow_components(labels)
    remove_small_components(labels, min_size=1000)
    return labels


def process_file(
    image_path: Path,
    sessions: Sequence[ort.InferenceSession],
    output_path: Path,
    model_input_shape: Tuple[int, int],
) -> None:
    """执行单张图片的推理与后处理。"""

    image = _load_image(image_path)
    model_input, pads, processed_shape = _prepare_model_input(image, model_input_shape)
    labels = _infer_labels(sessions, model_input, pads, processed_shape)

    if processed_shape != image.shape[:2]:
        labels = cv2.resize(
            labels.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    processed = process_label(labels)
    save_label_image(output_path, processed)


def gather_images(path: Path) -> Iterable[Path]:
    """收集需要处理的图像路径。"""

    if path.is_file():
        return [path]
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    # 使用 set 去重后再排序，避免相同后缀大小写造成的重复
    return sorted(set(files))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="对模型预测执行推理与后处理")
    parser.add_argument("--onnx_dir", type=Path, required=True, help="包含 ONNX 模型的文件或目录")
    parser.add_argument("--images_dir", type=Path, required=True, help="输入原始图像路径或目录")
    parser.add_argument("--output_dir", type=Path, required=True, help="输出目录")
    parser.add_argument(
        "--model_input_height",
        type=int,
        default=1024,
        help="ONNX 模型期望的高度 (默认: 1024)",
    )
    parser.add_argument(
        "--model_input_width",
        type=int,
        default=1024,
        help="ONNX 模型期望的宽度 (默认: 1024)",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口：根据输入类型批量或单张处理并保存。"""

    args = parse_args()

    sessions = _load_sessions(args.onnx_dir)
    images = list(gather_images(args.images_dir))
    if not images:
        raise FileNotFoundError("未找到任何输入图像")

    model_shape = (args.model_input_height, args.model_input_width)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        output_path = output_dir / (image_path.stem + ".png")
        process_file(image_path, sessions, output_path, model_shape)


if __name__ == "__main__":
    main()
