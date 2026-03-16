"""
OCR 引擎占位实现：未安装 PaddleOCR 时由 pipeline 使用。

仅返回空文字与默认方向，用于验证「分割 + 透视校正」流程。
安装 PaddleOCR 后 pipeline 会改用 waybill_ocr.ocr_engine.WaybillOCR。
"""

from __future__ import annotations

import numpy as np


class WaybillOCRStub:
    """占位 OCR：不调用 PaddleOCR，直接返回空结果。"""

    def recognize(self, image: np.ndarray) -> dict:
        """
        与 WaybillOCR.recognize 相同接口。

        Returns:
            full_text: 空字符串
            lines: 空列表
            orientation: 0
        """
        return {
            "full_text": "",
            "lines": [],
            "orientation": 0,
            "rotated_image": image,
        }
