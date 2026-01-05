"""图形化交互界面，基于 ``predict_auto_seg_demo.py`` 的两阶段分割流程。

功能特点：
- 顶部菜单提供“打开”选项，可选择本地图像自动运行两阶段分割。
- 仅在主窗口展示叠加环形光斑后的结果，不显示概率图或掩码。
- 侧边提供多种交互按钮：正向点击点、负向点击点、划线、框选。
  按钮可点击选中或取消；除正向点击外，其余按钮仅切换状态不执行功能。
- 自动选择可用的中文字体，避免界面中文显示为方框。
"""
from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox
from typing import Optional, Tuple

import cv2
import matplotlib
import numpy as np
import torch

from predict import draw_ring_circles, load_image, load_model, postprocess_mask, run_inference
from predict_auto_seg_demo import (
    FIRST_STAGE_INPUT,
    SECOND_STAGE_INPUT,
    _first_stage_inference,
    _iterative_second_stage,
    _map_point,
)
from prediction_postprecess_firststep import _load_sessions

# 保障中文字体正常显示
PREFERRED_FONTS = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["font.sans-serif"] = PREFERRED_FONTS + matplotlib.rcParams.get("font.sans-serif", [])
matplotlib.rcParams["axes.unicode_minus"] = False


def _apply_chinese_font() -> Optional[str]:
    """选取可用的中文字体并应用到 Tk 默认字体。"""
    available = set(tkfont.families())
    chosen = None
    for name in PREFERRED_FONTS:
        if name in available:
            chosen = name
            break
    if chosen is None:
        return None
    for family in ("TkDefaultFont", "TkMenuFont", "TkTextFont", "TkHeadingFont"):
        try:
            tkfont.nametofont(family).configure(family=chosen)
        except tk.TclError:
            continue
    return chosen


class SegmentationApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("两阶段光斑排布交互界面")

        chosen_font = _apply_chinese_font()
        if chosen_font:
            self.root.option_add("*Font", chosen_font)

        self.device = torch.device(args.device)
        self.sessions = None
        self.click_model = None
        self._load_models()

        self.resized_bgr: Optional[np.ndarray] = None
        self.image_tensor: Optional[torch.Tensor] = None
        self.reference_mask: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.display_scale: float = 1.0
        self._photo_image: Optional[tk.PhotoImage] = None

        self.active_mode: Optional[str] = None

        self._build_menu()
        self._build_layout()
        self._bind_canvas_events()

    def _load_models(self) -> None:
        self.sessions = _load_sessions(self.args.onnx_dir)
        self.click_model = load_model(str(self.args.click_model), self.device)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开", command=self._open_image)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_layout(self) -> None:
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, bg="#222", width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = tk.Frame(container, width=180)
        side.pack(side=tk.RIGHT, fill=tk.Y)

        instructions = tk.Label(side, text="选择操作后在图像上点击/拖动")
        instructions.pack(pady=(10, 5))

        self.buttons = {}
        actions = [
            ("positive", "正向点击点"),
            ("negative", "负向点击点"),
            ("line", "划线"),
            ("box", "框选"),
        ]
        for key, label in actions:
            btn = tk.Button(side, text=label, relief=tk.RAISED, command=lambda k=key: self._toggle_mode(k))
            btn.pack(fill=tk.X, padx=12, pady=4)
            self.buttons[key] = btn

        self.status_var = tk.StringVar(value="请通过文件菜单打开图像")
        status_label = tk.Label(side, textvariable=self.status_var, wraplength=160, justify=tk.LEFT)
        status_label.pack(padx=10, pady=(10, 5), anchor="w")

    def _bind_canvas_events(self) -> None:
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _toggle_mode(self, mode: str) -> None:
        if self.active_mode == mode:
            self.active_mode = None
        else:
            self.active_mode = mode
        for key, btn in self.buttons.items():
            if self.active_mode == key:
                btn.config(relief=tk.SUNKEN, bg="#a3d9ff")
            else:
                btn.config(relief=tk.RAISED, bg="SystemButtonFace")
        if self.active_mode:
            self._set_status(f"当前模式：{self.buttons[self.active_mode]['text']}")
        else:
            self._set_status("未选择模式")

    def _open_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="选择输入图像",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("所有文件", "*.*")],
        )
        if not filename:
            return
        try:
            self._load_image_and_run(Path(filename))
            self._set_status(f"已加载：{filename}")
        except Exception as exc:  # pragma: no cover - 运行时提示
            messagebox.showerror("加载失败", f"处理图像时出错：{exc}")

    def _load_image_and_run(self, image_path: Path) -> None:
        rgb_image, _processed_labels, green_center, red_mask = _first_stage_inference(
            image_path, self.sessions, FIRST_STAGE_INPUT
        )
        if green_center is None:
            fallback_center = self._centroid(red_mask)
            if fallback_center is None:
                raise RuntimeError("无法确定初始点击点，请检查第一阶段分割结果。")
            green_center = fallback_center

        resized_bgr, image_tensor = load_image(str(image_path), target_size=SECOND_STAGE_INPUT)
        mapped_click = _map_point(
            green_center, (rgb_image.shape[0], rgb_image.shape[1]), (SECOND_STAGE_INPUT[1], SECOND_STAGE_INPUT[0])
        )
        red_mask_resized = cv2.resize(red_mask.astype(np.uint8), SECOND_STAGE_INPUT, interpolation=cv2.INTER_NEAREST)

        _prob_map, final_mask, _final_center = _iterative_second_stage(
            self.click_model, self.device, image_tensor, mapped_click, self.args.iterations, self.args.threshold
        )

        self.resized_bgr = resized_bgr
        self.image_tensor = image_tensor
        self.reference_mask = red_mask_resized
        self.current_mask = final_mask.astype(np.uint8)
        self._refresh_overlay()

    @staticmethod
    def _centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return None
        y_mean, x_mean = coords.mean(axis=0)
        return int(round(x_mean)), int(round(y_mean))

    def _refresh_overlay(self) -> None:
        if self.resized_bgr is None or self.current_mask is None:
            return
        overlay = draw_ring_circles(self.resized_bgr, self.current_mask, reference_mask=self.reference_mask)
        self._update_canvas(overlay)

    def _update_canvas(self, bgr_image: np.ndarray) -> None:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        max_w, max_h = 1000, 900
        scale = min(max_w / w, max_h / h, 1.0)
        self.display_scale = scale
        if scale != 1.0:
            rgb_image = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        data = rgb_image.astype(np.uint8).tobytes()
        header = f"P6 {rgb_image.shape[1]} {rgb_image.shape[0]} 255\n".encode()
        self._photo_image = tk.PhotoImage(data=header + data)
        self.canvas.config(width=rgb_image.shape[1], height=rgb_image.shape[0])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_image)

    def _on_click(self, event: tk.Event) -> None:
        if self.resized_bgr is None or self.image_tensor is None or self.current_mask is None:
            self._set_status("请先打开图像")
            return
        if self.active_mode == "positive":
            self._handle_positive_click(event)
        # 其他模式仅展示可点击效果

    def _on_drag(self, event: tk.Event) -> None:
        # 预留扩展：目前划线、框选仅为 UI 切换状态。
        return

    def _on_release(self, event: tk.Event) -> None:
        # 当前不处理拖拽绘制产生的区域。
        return

    def _handle_positive_click(self, event: tk.Event) -> None:
        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)
        prob_map, binary_mask = run_inference(self.click_model, self.device, self.image_tensor, (x, y), self.args.threshold)
        processed_mask = postprocess_mask(binary_mask, (y, x))
        self.current_mask = processed_mask.astype(np.uint8)
        self._set_status(f"已根据点击点 ({x}, {y}) 更新结果")
        self._refresh_overlay()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="两阶段自动分割交互式界面")
    parser.add_argument("--onnx_dir", type=Path, default=Path(r"C:\work space\prp\predict_demo_260105\fold_1_checkpoint_best.onnx"), help="第一阶段 ONNX 模型目录或文件")
    parser.add_argument("--click_model", type=Path, default=Path(r"C:\work space\prp\predict_demo_260105\prp_segmenter.pth"), help="第二阶段点击分割模型权重路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="分割阈值")
    parser.add_argument("--iterations", type=int, default=3, help="自动迭代次数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备选择")
    return parser.parse_args()


if __name__ == "__main__":
    SegmentationApp(parse_args()).run()
