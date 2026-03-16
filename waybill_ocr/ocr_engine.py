"""
PaddleOCR 封装模块。

职责：
    1. 初始化 PaddleOCR（含方向分类器）
    2. 方向矫正：四方向旋转择优（处理 90°/270°）
    3. 执行 OCR 识别
    4. 结果排序（按阅读顺序：上到下、左到右）
    5. 格式化输出

依赖：paddlepaddle, paddleocr
"""

from __future__ import annotations

import cv2
import numpy as np

# 延迟导入，便于未安装时 pipeline 可回退到 stub
def _import_paddle_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR


def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """将图像旋转 angle 度（0/90/180/270）。"""
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"仅支持 0/90/180/270，当前 angle={angle}")


def _resize_to_max(image: np.ndarray, max_size: int) -> np.ndarray:
    """缩放到最长边为 max_size，保持比例。"""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


class WaybillOCR:
    """快递单 OCR：方向矫正 + PaddleOCR 识别 + 按行排序。"""

    def __init__(
        self,
        model_dir: str | None = None,
        use_gpu: bool = False,
        lang: str = "ch",
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.8,
        thumbnail_size: int = 640,
    ):
        PaddleOCR = _import_paddle_ocr()

        model_kwargs = {}
        if model_dir is not None:
            import os
            os.makedirs(model_dir, exist_ok=True)
            det_dir = os.path.join(model_dir, "det")
            rec_dir = os.path.join(model_dir, "rec")
            cls_dir = os.path.join(model_dir, "cls")
            os.makedirs(det_dir, exist_ok=True)
            os.makedirs(rec_dir, exist_ok=True)
            os.makedirs(cls_dir, exist_ok=True)
            model_kwargs["det_model_dir"] = det_dir
            model_kwargs["rec_model_dir"] = rec_dir
            model_kwargs["cls_model_dir"] = cls_dir

        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            show_log=False,
            **model_kwargs,
        )
        self.thumbnail_size = thumbnail_size

    def _score_orientation(self, image: np.ndarray, conf_threshold: float = 0.7) -> float:
        """对单个方向的图像执行 OCR 并计算得分（不使用 angle_cls，避免逐行翻转干扰）。"""
        small = _resize_to_max(image, self.thumbnail_size)
        result = self.ocr.ocr(small, cls=False)
        score = 0.0
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = float(line[1][1])
                if conf >= conf_threshold:
                    score += conf * len(text)
        return score

    def find_best_orientation(self, image: np.ndarray) -> int:
        """
        通过整图旋转对比，从 0°/90°/180° 中选择最佳方向。

        策略：
        1. 始终对比 0° 和 180°（整图翻转，不依赖 angle_cls 逐行判断）
        2. 若图片为横向（h < w×0.8），额外对比 90°
        3. 0° 作为先验最优方向，其他方向需超出 50% 才会替换
           （快递单绝大多数正向拍摄，高门槛避免噪声误判）

        评分时关闭 angle_cls（cls=False），确保纯粹对比文字方向的识别质量。
        """
        conf_threshold = 0.7
        # 0° 优先门槛：其他方向得分需超出 0° 的 50%
        prefer_0_ratio = 1.5

        score_0 = self._score_orientation(image, conf_threshold)
        best_angle, best_score = 0, score_0

        score_180 = self._score_orientation(
            _rotate_image(image, 180), conf_threshold)
        if score_180 > score_0 * prefer_0_ratio:
            best_angle, best_score = 180, score_180

        h, w = image.shape[:2]
        if h < w * 0.8:
            for angle in [90, 270]:
                score = self._score_orientation(
                    _rotate_image(image, angle), conf_threshold)
                if score > best_score * prefer_0_ratio:
                    best_angle, best_score = angle, score

        return best_angle

    def recognize(self, image: np.ndarray) -> dict:
        """
        输入透视校正后的图像，输出识别结果。

        Returns:
            full_text: 按阅读顺序拼接的全文（行间 \\n）
            lines: 按行排序的列表，每项含 text, confidence, box, center
            orientation: 选中的矫正角度（度）
            rotated_image: 方向矫正后的图像（若未旋转则与输入相同）
        """
        best_angle = self.find_best_orientation(image)
        if best_angle != 0:
            image = _rotate_image(image, best_angle)

        result = self.ocr.ocr(image, cls=False)

        lines = []
        if result and result[0]:
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                conf = float(line[1][1])
                center_y = sum(p[1] for p in box) / 4
                center_x = sum(p[0] for p in box) / 4
                lines.append({
                    "text": text,
                    "confidence": conf,
                    "box": box,
                    "center": (center_x, center_y),
                })

            lines.sort(key=lambda l: (
                round(l["center"][1] / 20) * 20,
                l["center"][0],
            ))

        return {
            "full_text": "\n".join(l["text"] for l in lines),
            "lines": lines,
            "orientation": best_angle,
            "rotated_image": image,
        }
