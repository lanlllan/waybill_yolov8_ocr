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
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou

    def segment(self, image: np.ndarray) -> list[dict]:
        """
        对一张图片进行分割，返回所有检测到的快递单的掩码与检测信息。

        Args:
            image: BGR 图像 (H, W, 3)，numpy uint8。

        Returns:
            列表，每项为：
            - mask: 二值掩码 (H, W)，uint8，255 为前景
            - bbox: [x1, y1, x2, y2] 检测框
            - confidence: 置信度
            - class_id: 类别 ID
            - class_name: 类别名（如 waybill）
        """
        results = self.model.predict(
            image,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        detections = []
        h, w = image.shape[:2]

        for r in results:
            if r.masks is None:
                continue
            names = r.names or {}
            for j, mask_tensor in enumerate(r.masks.data):
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

                box = r.boxes[j]
                bbox = box.xyxy.cpu().numpy().flatten()  # shape (4,) 供 pipeline 使用 .tolist()
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

        return detections
