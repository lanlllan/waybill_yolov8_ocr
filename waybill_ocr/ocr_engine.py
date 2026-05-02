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
import time

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


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


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
        orientation_conf_threshold: float = 0.7,
        orientation_early_accept_0: bool = False,
        orientation_high_conf_threshold: float = 0.9,
        orientation_accept_min_lines: int = 8,
        orientation_accept_min_score: float = 80.0,
        orientation_prefer_0_ratio: float = 1.5,
        orientation_candidate_angles: list[int] | None = None,
        line_sort_y_bucket: int = 20,
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
        self.orientation_conf_threshold = orientation_conf_threshold
        self.orientation_early_accept_0 = orientation_early_accept_0
        self.orientation_high_conf_threshold = orientation_high_conf_threshold
        self.orientation_accept_min_lines = orientation_accept_min_lines
        self.orientation_accept_min_score = orientation_accept_min_score
        self.orientation_prefer_0_ratio = orientation_prefer_0_ratio
        self.orientation_candidate_angles = orientation_candidate_angles or [180, 90, 270]
        self.line_sort_y_bucket = max(1, int(line_sort_y_bucket))

    def _score_orientation(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.7,
        high_conf_threshold: float | None = None,
        return_stats: bool = False,
    ):
        """对单个方向的图像执行 OCR 并计算得分（不使用 angle_cls，避免逐行翻转干扰）。"""
        small = _resize_to_max(image, self.thumbnail_size)
        result = self.ocr.ocr(small, cls=False)
        score = 0.0
        line_count = 0
        high_conf_count = 0
        high_conf_score = 0.0
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = float(line[1][1])
                if conf >= conf_threshold:
                    weighted = conf * len(text)
                    score += weighted
                    line_count += 1
                    if high_conf_threshold is not None and conf >= high_conf_threshold:
                        high_conf_count += 1
                        high_conf_score += weighted
        if return_stats:
            return {
                "score": score,
                "line_count": line_count,
                "high_conf_count": high_conf_count,
                "high_conf_score": high_conf_score,
            }
        return score

    def find_best_orientation(self, image: np.ndarray, return_details: bool = False):
        """
        通过整图旋转对比，从 0°/90°/180°/270° 中选择最佳方向。

        策略：
        1. 评估 0°、180°、90°、270° 四个候选方向
        2. 0° 作为先验方向，非 0° 候选需超过 0° 得分的 1.5 倍才参与竞争
        3. 在满足阈值的候选方向中选择得分最高者，避免候选顺序影响最终结果

        评分时关闭 angle_cls（cls=False），确保纯粹对比文字方向的识别质量。
        """
        conf_threshold = self.orientation_conf_threshold
        high_conf_threshold = self.orientation_high_conf_threshold
        prefer_0_ratio = self.orientation_prefer_0_ratio

        details = []

        t0 = time.perf_counter()
        stats_0 = self._score_orientation(
            image,
            conf_threshold,
            high_conf_threshold=high_conf_threshold,
            return_stats=True,
        )
        score_0 = stats_0["score"]
        details.append({
            "angle": 0,
            "score": round(float(score_0), 4),
            "line_count": stats_0["line_count"],
            "high_conf_count": stats_0["high_conf_count"],
            "high_conf_score": round(float(stats_0["high_conf_score"]), 4),
            "elapsed_ms": _elapsed_ms(t0),
        })
        best_angle, best_score = 0, score_0
        switch_threshold = score_0 * prefer_0_ratio

        early_accepted = (
            self.orientation_early_accept_0
            and stats_0["high_conf_count"] >= self.orientation_accept_min_lines
            and stats_0["high_conf_score"] >= self.orientation_accept_min_score
        )
        if early_accepted:
            if return_details:
                return best_angle, {
                    "selected_angle": best_angle,
                    "switch_threshold": round(float(switch_threshold), 4),
                    "early_accepted": True,
                    "early_accept_reason": "0deg_high_conf_text",
                    "candidates": details,
                    "total_ms": round(sum(d["elapsed_ms"] for d in details), 2),
                }
            return best_angle

        for angle in self.orientation_candidate_angles:
            if angle == 0:
                continue
            t_angle = time.perf_counter()
            stats = self._score_orientation(
                _rotate_image(image, angle),
                conf_threshold,
                high_conf_threshold=high_conf_threshold,
                return_stats=True,
            )
            score = stats["score"]
            details.append({
                "angle": angle,
                "score": round(float(score), 4),
                "line_count": stats["line_count"],
                "high_conf_count": stats["high_conf_count"],
                "high_conf_score": round(float(stats["high_conf_score"]), 4),
                "elapsed_ms": _elapsed_ms(t_angle),
            })
            if score > switch_threshold and score > best_score:
                best_angle, best_score = angle, score

        if return_details:
            return best_angle, {
                "selected_angle": best_angle,
                "switch_threshold": round(float(switch_threshold), 4),
                "early_accepted": False,
                "candidates": details,
                "total_ms": round(sum(d["elapsed_ms"] for d in details), 2),
            }
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
        t_total = time.perf_counter()
        t_orientation = time.perf_counter()
        best_angle, orientation_timing = self.find_best_orientation(
            image, return_details=True)
        orientation_timing["total_ms"] = _elapsed_ms(t_orientation)
        if best_angle != 0:
            image = _rotate_image(image, best_angle)

        t_final_ocr = time.perf_counter()
        result = self.ocr.ocr(image, cls=False)
        final_ocr_ms = _elapsed_ms(t_final_ocr)

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
                round(l["center"][1] / self.line_sort_y_bucket) * self.line_sort_y_bucket,
                l["center"][0],
            ))

        return {
            "full_text": "\n".join(l["text"] for l in lines),
            "lines": lines,
            "orientation": best_angle,
            "rotated_image": image,
            "timing": {
                "orientation_ms": orientation_timing["total_ms"],
                "orientation_selected_angle": orientation_timing["selected_angle"],
                "orientation_switch_threshold": orientation_timing["switch_threshold"],
                "orientation_early_accepted": orientation_timing.get("early_accepted", False),
                "orientation_early_accept_reason": orientation_timing.get("early_accept_reason", ""),
                "orientation_detail": orientation_timing["candidates"],
                "final_ocr_ms": final_ocr_ms,
                "ocr_total_ms": _elapsed_ms(t_total),
            },
        }
