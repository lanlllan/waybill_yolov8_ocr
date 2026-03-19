"""
主流程模块：串联 segmentor → rectifier → ocr_engine，支持单图和批量处理。

输出目录结构（每张图一个子文件夹）：
    output/
    ├── 206-0/
    │   ├── 0_rectified.jpg      # 透视校正后的快递单图片
    │   ├── 0_process.jpg        # 矫正过程可视化（各阶段中间结果拼接）
    │   ├── 0_ocr_boxes.jpg      # OCR 文本检测框可视化（框+文字+置信度）
    │   ├── 0_ocr.txt            # OCR 提取的纯文本
    │   ├── 0_result.json        # 完整结果（含置信度、坐标等）
    │   └── ...                  # 多个快递单时依次编号
    ├── 207-0/
    │   └── ...
    └── results.json             # 汇总 JSON（可选，--json 开启）
"""

from __future__ import annotations

import json
import logging
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
    PREPROCESSING_ENABLED,
    PREPROCESSING_MODE,
    PREPROCESSING_ENHANCE_CONTRAST,
    PREPROCESSING_DENOISE,
    PREPROCESSING_SHARPEN,
    PREPROCESSING_ADJUST_BRIGHTNESS,
    PREPROCESSING_BINARIZE,
    PREPROCESSING_CONTRAST_CLIP_LIMIT,
    PREPROCESSING_DENOISE_STRENGTH,
    PREPROCESSING_SHARPEN_STRENGTH,
    PREPROCESSING_TARGET_BRIGHTNESS,
)
import numpy as np
from waybill_ocr.segmentor import WaybillSegmentor
from waybill_ocr.rectifier import (
    rectify_from_mask, draw_quad_on_image, draw_mask_overlay,
)
from waybill_ocr.preprocessing import preprocess_for_ocr, auto_preprocess

logger = logging.getLogger(__name__)


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
    except ImportError:
        logger.warning("PaddleOCR 未安装，使用空 Stub 引擎（OCR 结果将为空）")
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
        """在图片顶部添加自适应大小的标签栏。"""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        max_scale = 0.8
        min_scale = 0.3

        # 从大到小尝试字号，找到能放进面板宽度的最大字号
        scale = max_scale
        while scale >= min_scale:
            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
            if text_size[0] <= w - 10:
                break
            scale -= 0.05
        else:
            text_size = cv2.getTextSize(text, font, min_scale, thickness)[0]
            scale = min_scale

        bar_h = text_size[1] + 20
        bar = np.full((bar_h, w, 3), 40, dtype=np.uint8)
        x = max((w - text_size[0]) // 2, 4)
        y = text_size[1] + 10
        cv2.putText(bar, text, (x, y), font, scale, (255, 255, 255), thickness)
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

    @staticmethod
    def _get_cjk_font(size: int = 16):
        """加载中文字体，用于 Pillow 绘制。"""
        from PIL import ImageFont
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    @staticmethod
    def _draw_ocr_boxes(image: np.ndarray, lines: list) -> np.ndarray:
        """在图像上绘制 OCR 检测框、识别文字和置信度（支持中文）。"""
        from PIL import Image, ImageDraw

        vis = image.copy()
        # 先用 OpenCV 画多边形框
        for line in lines:
            box = line.get("box")
            conf = line.get("confidence", 0)
            if not box or len(box) < 4:
                continue
            pts = np.array(box, dtype=np.int32)
            color = (0, 200, 0) if conf >= 0.9 else (0, 200, 255) if conf >= 0.7 else (0, 0, 255)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

        # 转 PIL 绘制中文文字
        pil_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        h_img = vis.shape[0]
        font_size = max(12, min(20, h_img // 50))
        font = WaybillPipeline._get_cjk_font(font_size)

        for line in lines:
            box = line.get("box")
            text = line.get("text", "")
            conf = line.get("confidence", 0)
            if not box or len(box) < 4:
                continue

            x = int(box[0][0])
            y = int(box[0][1]) - font_size - 4
            if y < 0:
                y = int(box[2][1]) + 2
            label = f"{text} ({conf:.2f})"
            color_rgb = (0, 200, 0) if conf >= 0.9 else (0, 200, 255) if conf >= 0.7 else (255, 0, 0)

            bbox_text = draw.textbbox((x, y), label, font=font)
            draw.rectangle(bbox_text, fill=(0, 0, 0))
            draw.text((x, y), label, fill=color_rgb, font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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

                # 应用图像预处理以提升 OCR 质量
                if PREPROCESSING_ENABLED and PREPROCESSING_MODE != "off":
                    if PREPROCESSING_MODE == "auto":
                        # 自动模式：根据图像质量自动选择预处理方法
                        rectified = auto_preprocess(rectified)
                    else:
                        # 自定义模式：使用配置文件中的参数
                        rectified = preprocess_for_ocr(
                            rectified,
                            enable_contrast=PREPROCESSING_ENHANCE_CONTRAST,
                            enable_denoise=PREPROCESSING_DENOISE,
                            enable_sharpen=PREPROCESSING_SHARPEN,
                            enable_brightness=PREPROCESSING_ADJUST_BRIGHTNESS,
                            enable_binarize=PREPROCESSING_BINARIZE,
                            contrast_clip_limit=PREPROCESSING_CONTRAST_CLIP_LIMIT,
                            denoise_strength=PREPROCESSING_DENOISE_STRENGTH,
                            sharpen_strength=PREPROCESSING_SHARPEN_STRENGTH,
                            target_brightness=PREPROCESSING_TARGET_BRIGHTNESS,
                        )
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

            try:
                ocr_result = self.ocr.recognize(rectified)
            except Exception as e:
                logger.warning("OCR 识别失败 (目标 #%d): %s", i, e)
                item = {
                    "index": i,
                    "class": det.get("class_name", "waybill"),
                    "confidence": float(det.get("confidence", 0)),
                    "bbox": det["bbox"].tolist() if hasattr(det["bbox"], "tolist") else list(det["bbox"]),
                    "error": f"OCR 识别失败: {e}",
                    "text": "",
                    "lines": [],
                    "orientation": 0,
                }
                results.append(item)
                self._save_result(sub_dir, i, item, rectified=rectified if SAVE_DEBUG_IMAGES else None)
                continue

            orientation = ocr_result.get("orientation", 0)
            final_image = ocr_result.get("rotated_image", rectified)

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
                rectified=final_image if SAVE_DEBUG_IMAGES else None,
            )

            if SAVE_DEBUG_IMAGES:
                try:
                    process_img = self._build_process_image(
                        image, det, rect_result, final_image, orientation)
                    process_path = os.path.join(sub_dir, f"{i}_process.jpg")
                    cv2.imwrite(process_path, process_img)
                except Exception:
                    logger.warning("调试图生成失败 (目标 #%d)", i, exc_info=True)

                if ocr_result.get("lines"):
                    try:
                        ocr_vis = self._draw_ocr_boxes(final_image, ocr_result["lines"])
                        ocr_vis_path = os.path.join(sub_dir, f"{i}_ocr_boxes.jpg")
                        cv2.imwrite(ocr_vis_path, ocr_vis)
                    except Exception:
                        logger.warning("OCR 可视化图生成失败 (目标 #%d)", i, exc_info=True)

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
        actual_dir = os.path.join(args.output, stem)
        print(f"\n=== {path} → {actual_dir}/ ===")
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
