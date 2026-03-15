"""
YOLO 分割模块。

职责：加载 YOLOv8-Seg 模型（支持 .pt / .onnx），输入图片，输出每个检测目标的掩码。
模型路径由 config 决定：优先 waybill_ocr/models/yolo/best.onnx，否则 export/export5/best.onnx。
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO


class WaybillSegmentor:
    """使用 YOLOv8-Seg 对图片中的快递单进行检测与分割，返回掩码与检测框。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf: float = 0.5,
        iou: float = 0.7,
        imgsz: int = 640,
    ):
        self.model = YOLO(model_path, task="segment")
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def segment(self, image: np.ndarray) -> tuple[list[dict], np.ndarray | None]:
        """
        对一张图片进行分割，返回检测结果列表和 YOLO 标注图。

        Args:
            image: BGR 图像 (H, W, 3)，numpy uint8。

        Returns:
            (detections, annotated_image) 元组：
            detections — 列表，每项含：
                - mask: 二值掩码 (H, W)，uint8，255 为前景
                - bbox: [x1, y1, x2, y2] 检测框
                - confidence: 置信度
                - class_id: 类别 ID
                - class_name: 类别名（如 waybill）
            annotated_image — YOLO 绘制的标注图（含掩码+bbox+标签），
                             无检测结果时为 None。
        """
        results = self.model.predict(
            image,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        annotated = None
        h, w = image.shape[:2]

        for r in results:
            if annotated is None:
                annotated = r.plot()

            if r.masks is None:
                continue
            names = r.names or {}
            polygons = r.masks.xy if hasattr(r.masks, "xy") and r.masks.xy is not None else None

            for j in range(len(r.masks.data)):
                if polygons is not None and j < len(polygons) and len(polygons[j]) >= 3:
                    poly = polygons[j].astype(np.int32).reshape(-1, 1, 2)
                    binary_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(binary_mask, [poly], 255)
                else:
                    mask_np = r.masks.data[j].cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
                    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

                mask_area = np.count_nonzero(binary_mask)
                min_mask_area = h * w * 0.005
                if mask_area < min_mask_area:
                    continue

                box = r.boxes[j]
                bbox = box.xyxy.cpu().numpy().flatten()
                conf = float(box.conf.cpu().numpy().item())
                cls_id = int(box.cls.cpu().numpy().item())
                cls_name = names.get(cls_id, "waybill")

                detections.append({
                    "mask": binary_mask,
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        return detections, annotated
