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
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.8,
        thumbnail_size: int = 640,
    ):
        PaddleOCR = _import_paddle_ocr()
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            show_log=False,
        )
        self.thumbnail_size = thumbnail_size

    def find_best_orientation(self, image: np.ndarray) -> int:
        """
        判断是否需要旋转 90°/270°，返回最佳角度。

        PaddleOCR 的 angle_cls 已能自动处理 0°/180° 翻转，
        因此只在图片为横向时额外测试 90°/270°（竖版快递单横拍的情况）。

        评分策略：
        - 对比 0° 和 90° 的识别结果（高置信度框的 conf×len 总分）
        - 90° 需超出 0° 至少 20% 才选 90°（0° 优先原则）
        """
        h, w = image.shape[:2]

        # 竖向或接近正方形的图，0°/180° 交给 angle_cls 处理即可
        if h >= w * 0.8:
            return 0

        # 横向图：对比 0° 和 90°
        conf_threshold = 0.7
        scores = {}

        for angle in [0, 90]:
            rotated = _rotate_image(image, angle)
            small = _resize_to_max(rotated, self.thumbnail_size)
            result = self.ocr.ocr(small, cls=True)

            score = 0.0
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    conf = float(line[1][1])
                    if conf >= conf_threshold:
                        score += conf * len(text)
            scores[angle] = score

        if scores[90] > scores[0] * 1.2:
            return 90
        return 0

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
