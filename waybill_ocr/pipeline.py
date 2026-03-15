"""
主流程模块：串联 segmentor → rectifier → ocr_engine，支持单图和批量处理。

输出目录结构（每张图一个子文件夹）：
    output/
    ├── 206-0/
    │   ├── 0_rectified.jpg      # 透视校正后的快递单图片
    │   ├── 0_process.jpg        # 矫正过程可视化（各阶段中间结果拼接）
    │   ├── 0_ocr.txt            # OCR 提取的纯文本
    │   ├── 0_result.json        # 完整结果（含置信度、坐标等）
    │   └── ...                  # 多个快递单时依次编号
    ├── 207-0/
    │   └── ...
    └── results.json             # 汇总 JSON（可选，--json 开启）
"""

from __future__ import annotations

import json
import os
import argparse
from datetime import datetime

import cv2
from waybill_ocr.config import (
    YOLO_MODEL_PATH,
    YOLO_DEVICE,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_IMGSZ,
    SAVE_DEBUG_IMAGES,
    OUTPUT_DIR,
    OCR_MODEL_DIR,
    OCR_USE_GPU,
    OCR_LANG,
    OCR_USE_ANGLE_CLS,
    OCR_DET_DB_THRESH,
    OCR_DET_DB_BOX_THRESH,
    OCR_DET_DB_UNCLIP_RATIO,
    ORIENTATION_THUMBNAIL_SIZE,
)
import numpy as np
from waybill_ocr.segmentor import WaybillSegmentor
from waybill_ocr.rectifier import (
    rectify_from_mask, draw_quad_on_image, draw_mask_overlay,
)


def _get_ocr_engine():
    """延迟导入 PaddleOCR，未安装时返回占位实现。"""
    try:
        from waybill_ocr.ocr_engine import WaybillOCR
        return WaybillOCR(
            model_dir=OCR_MODEL_DIR,
            use_gpu=OCR_USE_GPU,
            lang=OCR_LANG,
            use_angle_cls=OCR_USE_ANGLE_CLS,
            det_db_thresh=OCR_DET_DB_THRESH,
            det_db_box_thresh=OCR_DET_DB_BOX_THRESH,
            det_db_unclip_ratio=OCR_DET_DB_UNCLIP_RATIO,
            thumbnail_size=ORIENTATION_THUMBNAIL_SIZE,
        )
    except Exception:
        from waybill_ocr.ocr_engine_stub import WaybillOCRStub
        return WaybillOCRStub()


def _make_serializable(obj):
    """递归将 numpy 等不可序列化的对象转为 Python 原生类型。"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


class WaybillPipeline:
    """快递单 OCR 全流程：分割 → 透视校正 → 方向矫正 → OCR 识别。"""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.segmentor = WaybillSegmentor(
            YOLO_MODEL_PATH,
            device=YOLO_DEVICE,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            imgsz=YOLO_IMGSZ,
        )
        self.ocr = _get_ocr_engine()
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_image_output_dir(self, image_path: str) -> str:
        """为每张输入图片创建独立的输出子文件夹。"""
        stem = os.path.splitext(os.path.basename(image_path))[0]
        sub_dir = os.path.join(self.output_dir, stem)
        os.makedirs(sub_dir, exist_ok=True)
        return sub_dir

    @staticmethod
    def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        """等比例缩放图片到指定高度。"""
        h, w = img.shape[:2]
        if h == target_h:
            return img
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _add_label(img: np.ndarray, text: str) -> np.ndarray:
        """在图片顶部添加中文/英文标签栏。"""
        h, w = img.shape[:2]
        bar_h = 40
        bar = np.full((bar_h, w, 3), 40, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        x = (w - text_size[0]) // 2
        cv2.putText(bar, text, (x, 28), font, 0.7, (255, 255, 255), 2)
        return np.vstack([bar, img])

    def _build_process_image(self, image: np.ndarray, det: dict,
                             rect_result: dict, rectified: np.ndarray,
                             orientation: int) -> np.ndarray:
        """
        生成矫正过程可视化图：横向拼接各阶段中间结果。

        阶段：原图ROI → 掩码叠加 → 四角点检测 → 透视校正 → 方向矫正
        """
        bbox = det["bbox"]
        mask = det["mask"]
        src_pts = rect_result["src_pts"]
        cleaned_mask = rect_result["cleaned_mask"]

        bx1, by1, bx2, by2 = [int(v) for v in bbox[:4]]
        pad = 50
        h_img, w_img = image.shape[:2]
        rx1 = max(bx1 - pad, 0)
        ry1 = max(by1 - pad, 0)
        rx2 = min(bx2 + pad, w_img)
        ry2 = min(by2 + pad, h_img)

        # 1. 原图 ROI 区域
        roi = image[ry1:ry2, rx1:rx2].copy()
        cv2.rectangle(roi, (bx1 - rx1, by1 - ry1), (bx2 - rx1, by2 - ry1), (0, 0, 255), 3)

        # 2. 掩码叠加
        mask_vis = draw_mask_overlay(image[ry1:ry2, rx1:rx2], cleaned_mask[ry1:ry2, rx1:rx2])

        # 3. 四角点检测
        quad_vis = draw_quad_on_image(image[ry1:ry2, rx1:rx2],
                                      src_pts - np.array([rx1, ry1], dtype=np.float32),
                                      thickness=3)

        # 4. 透视校正结果
        rect_vis = rect_result["rectified"].copy()

        # 5. 方向矫正后（最终结果）
        final_vis = rectified.copy()

        target_h = 400
        stages = [
            (roi, "1. Input ROI"),
            (mask_vis, "2. Mask"),
            (quad_vis, "3. Quad Detection"),
            (rect_vis, "4. Perspective Transform"),
        ]
        if orientation != 0:
            stages.append((final_vis, f"5. Rotate {orientation} deg"))
        else:
            stages.append((final_vis, "5. Final (0 deg)"))

        panels = []
        for img, label in stages:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            resized = self._resize_to_height(img, target_h)
            labeled = self._add_label(resized, label)
            panels.append(labeled)

        # 面板之间加 2px 分隔线
        sep = np.full((panels[0].shape[0], 2, 3), 80, dtype=np.uint8)
        parts = []
        for i, p in enumerate(panels):
            if i > 0:
                h_p = p.shape[0]
                h_s = sep.shape[0]
                if h_p != h_s:
                    sep = np.full((h_p, 2, 3), 80, dtype=np.uint8)
                parts.append(sep)
            parts.append(p)

        # 统一高度
        max_h = max(p.shape[0] for p in parts)
        aligned = []
        for p in parts:
            if p.shape[0] < max_h:
                pad_bottom = np.full((max_h - p.shape[0], p.shape[1], 3), 40, dtype=np.uint8)
                p = np.vstack([p, pad_bottom])
            aligned.append(p)

        return np.hstack(aligned)

    def _save_result(self, sub_dir: str, index: int, result: dict,
                     rectified=None):
        """保存单个快递单的结果文件。"""
        prefix = f"{index}"

        if rectified is not None:
            img_path = os.path.join(sub_dir, f"{prefix}_rectified.jpg")
            cv2.imwrite(img_path, rectified)

        txt_path = os.path.join(sub_dir, f"{prefix}_ocr.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result.get("text", ""))

        json_path = os.path.join(sub_dir, f"{prefix}_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(result), f,
                      ensure_ascii=False, indent=2)

    def process_image(self, image_path: str) -> list[dict]:
        """处理单张图片，返回所有检测到的快递单的 OCR 结果。"""
        image = cv2.imread(image_path)
        if image is None:
            return [{"error": f"无法读取图片: {image_path}"}]

        sub_dir = self._get_image_output_dir(image_path)
        detections, annotated = self.segmentor.segment(image)
        results = []

        if annotated is not None and SAVE_DEBUG_IMAGES:
            yolo_path = os.path.join(sub_dir, "yolo_annotated.jpg")
            cv2.imwrite(yolo_path, annotated)

        for i, det in enumerate(detections):
            try:
                rect_result = rectify_from_mask(image, det["mask"], bbox=det["bbox"])
                rectified = rect_result["rectified"]
            except Exception as e:
                item = {
                    "index": i,
                    "class": det.get("class_name", "waybill"),
                    "confidence": det.get("confidence", 0),
                    "bbox": det.get("bbox", []),
                    "error": f"透视校正失败: {e}",
                    "text": "",
                    "lines": [],
                    "orientation": 0,
                }
                results.append(item)
                self._save_result(sub_dir, i, item)
                continue

            ocr_result = self.ocr.recognize(rectified)
            orientation = ocr_result.get("orientation", 0)

            item = {
                "index": i,
                "class": det["class_name"],
                "confidence": float(det["confidence"]),
                "bbox": det["bbox"].tolist() if hasattr(det["bbox"], "tolist") else list(det["bbox"]),
                "text": ocr_result["full_text"],
                "lines": ocr_result["lines"],
                "orientation": orientation,
            }
            results.append(item)

            self._save_result(
                sub_dir, i, item,
                rectified=rectified if SAVE_DEBUG_IMAGES else None,
            )

            if SAVE_DEBUG_IMAGES:
                try:
                    process_img = self._build_process_image(
                        image, det, rect_result, rectified, orientation)
                    process_path = os.path.join(sub_dir, f"{i}_process.jpg")
                    cv2.imwrite(process_path, process_img)
                except Exception:
                    pass

        return results

    def process_batch(self, image_paths: list[str]) -> dict[str, list[dict]]:
        """批量处理多张图片。"""
        all_results = {}
        for path in image_paths:
            all_results[path] = self.process_image(path)
        return all_results


def main():
    parser = argparse.ArgumentParser(description="快递单 OCR 提取")
    parser.add_argument("images", nargs="+", help="图片路径")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--json", "-j", action="store_true", help="保存汇总 results.json")
    args = parser.parse_args()

    pipeline = WaybillPipeline(output_dir=args.output)
    results = pipeline.process_batch(args.images)

    for path, items in results.items():
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== {path} → output/{stem}/ ===")
        for item in items:
            if "error" in item:
                print(f"  错误: {item['error']}")
                continue
            print(f"  快递单 #{item['index']} "
                  f"(置信度: {item['confidence']:.2f}, "
                  f"方向: {item['orientation']}°)")
            print(f"  {item['text']}")

    if args.json:
        json_path = os.path.join(args.output, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(results), f,
                      ensure_ascii=False, indent=2)
        print(f"\n汇总已保存: {json_path}")


if __name__ == "__main__":
    main()
