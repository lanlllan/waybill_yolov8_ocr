"""
主流程模块：串联 segmentor → rectifier → ocr_engine，支持单图和批量处理。
"""

from __future__ import annotations

import json
import os
import argparse
import cv2
from waybill_ocr.config import (
    YOLO_MODEL_PATH,
    YOLO_DEVICE,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    SAVE_DEBUG_IMAGES,
    OUTPUT_DIR,
    OCR_USE_GPU,
    OCR_LANG,
    OCR_USE_ANGLE_CLS,
    OCR_DET_DB_THRESH,
    ORIENTATION_THUMBNAIL_SIZE,
)
from waybill_ocr.segmentor import WaybillSegmentor
from waybill_ocr.rectifier import rectify_from_mask


def _get_ocr_engine():
    """延迟导入 PaddleOCR，未安装时返回占位实现。"""
    try:
        from waybill_ocr.ocr_engine import WaybillOCR
        return WaybillOCR(
            use_gpu=OCR_USE_GPU,
            lang=OCR_LANG,
            use_angle_cls=OCR_USE_ANGLE_CLS,
            det_db_thresh=OCR_DET_DB_THRESH,
            thumbnail_size=ORIENTATION_THUMBNAIL_SIZE,
        )
    except Exception:
        from waybill_ocr.ocr_engine_stub import WaybillOCRStub
        return WaybillOCRStub()


class WaybillPipeline:
    """快递单 OCR 全流程：分割 → 透视校正 → 方向矫正 → OCR 识别。"""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.segmentor = WaybillSegmentor(
            YOLO_MODEL_PATH,
            device=YOLO_DEVICE,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
        )
        self.ocr = _get_ocr_engine()
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image(self, image_path: str) -> list[dict]:
        """处理单张图片，返回所有检测到的快递单的 OCR 结果。"""
        image = cv2.imread(image_path)
        if image is None:
            return [{"error": f"无法读取图片: {image_path}"}]

        detections = self.segmentor.segment(image)
        results = []

        for i, det in enumerate(detections):
            try:
                rect_result = rectify_from_mask(image, det["mask"])
                rectified = rect_result["rectified"]
            except Exception as e:
                results.append({
                    "index": i,
                    "class": det.get("class_name", "waybill"),
                    "confidence": det.get("confidence", 0),
                    "bbox": det.get("bbox", []),
                    "error": f"透视校正失败: {e}",
                    "text": "",
                    "lines": [],
                    "orientation": 0,
                })
                continue

            ocr_result = self.ocr.recognize(rectified)

            results.append({
                "index": i,
                "class": det["class_name"],
                "confidence": float(det["confidence"]),
                "bbox": det["bbox"].tolist() if hasattr(det["bbox"], "tolist") else list(det["bbox"]),
                "text": ocr_result["full_text"],
                "lines": ocr_result["lines"],
                "orientation": ocr_result.get("orientation", 0),
            })

            if SAVE_DEBUG_IMAGES:
                out_path = os.path.join(
                    self.output_dir,
                    f"debug_{os.path.basename(image_path)}_{i}_rectified.jpg",
                )
                cv2.imwrite(out_path, rectified)

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
    parser.add_argument("--json", "-j", action="store_true", help="保存 results.json")
    args = parser.parse_args()

    pipeline = WaybillPipeline(output_dir=args.output)
    results = pipeline.process_batch(args.images)

    for path, items in results.items():
        print(f"\n=== {path} ===")
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
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n已保存: {json_path}")


if __name__ == "__main__":
    main()
