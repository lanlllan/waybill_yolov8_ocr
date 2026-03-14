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
        use_gpu: bool = False,
        lang: str = "ch",
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
        thumbnail_size: int = 640,
    ):
        PaddleOCR = _import_paddle_ocr()
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            det_db_thresh=det_db_thresh,
            show_log=False,
        )
        self.thumbnail_size = thumbnail_size

    def find_best_orientation(self, image: np.ndarray) -> int:
        """
        对 0°/90°/180°/270° 择优，返回最佳角度。
        竖向图只试 0° 和 180°，横向图试四方向。
        """
        h, w = image.shape[:2]
        if h > w:
            candidates = [0, 180]
        else:
            candidates = [0, 90, 180, 270]

        best_angle = 0
        best_score = -1.0

        for angle in candidates:
            rotated = _rotate_image(image, angle)
            small = _resize_to_max(rotated, self.thumbnail_size)
            result = self.ocr.ocr(small, cls=True)

            if result and result[0]:
                confs = [line[1][1] for line in result[0]]
                score = len(confs) * (sum(confs) / len(confs))
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_angle = angle

        return best_angle

    def recognize(self, image: np.ndarray) -> dict:
        """
        输入透视校正后的图像，输出识别结果。

        Returns:
            full_text: 按阅读顺序拼接的全文（行间 \\n）
            lines: 按行排序的列表，每项含 text, confidence, box, center
            orientation: 选中的矫正角度（度）
        """
        best_angle = self.find_best_orientation(image)
        if best_angle != 0:
            image = _rotate_image(image, best_angle)

        result = self.ocr.ocr(image, cls=True)

        lines = []
        if result and result[0]:
            for line in result[0]:
                box = line[0]  # 4 个角点 [[x,y], ...]
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

            # 按 y 量化到 20px 分行，同行按 x 排序
            lines.sort(key=lambda l: (
                round(l["center"][1] / 20) * 20,
                l["center"][0],
            ))

        return {
            "full_text": "\n".join(l["text"] for l in lines),
            "lines": lines,
            "orientation": best_angle,
        }
