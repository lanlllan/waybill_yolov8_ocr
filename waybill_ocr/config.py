"""
集中管理所有可配置参数。

路径基于项目根目录 waybill_ocr/（即本文件的上两级目录）。
"""

import os

# waybill_ocr/ 项目根目录（本文件位于 waybill_ocr/waybill_ocr/config.py）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# YOLO 分割模型配置
# ============================================================

YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo", "best.onnx")
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.7
YOLO_DEVICE = "cpu"  # "cpu" 或 "0"(GPU)

# ============================================================
# 透视校正配置（已调优）
# ============================================================

RECTIFIER_EPSILON_RATIO = 0.02
RECTIFIER_USE_CONVEX_HULL = True
RECTIFIER_MORPH_SIZE = 7

# ============================================================
# PaddleOCR 配置
# ============================================================

OCR_USE_ANGLE_CLS = True
OCR_LANG = "ch"
OCR_USE_GPU = False
OCR_DET_DB_THRESH = 0.3

ORIENTATION_THUMBNAIL_SIZE = 640

# ============================================================
# 输出配置
# ============================================================

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SAVE_DEBUG_IMAGES = True
