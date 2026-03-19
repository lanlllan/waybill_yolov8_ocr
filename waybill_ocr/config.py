"""
集中管理所有可配置参数。

加载顺序（后者覆盖前者）：
    1. config/default.yaml   — 默认配置（随仓库提交）
    2. config/local.yaml     — 本地覆盖（不提交，.gitignore 已忽略）

路径字段（如 model_path、dir）自动基于 PROJECT_ROOT 解析为绝对路径。
"""

import os
from pathlib import Path

import yaml

# waybill_ocr/ 项目根目录（本文件位于 waybill_ocr/waybill_ocr/config.py）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CONFIG_DIR = PROJECT_ROOT / "config"
_DEFAULT_CFG = _CONFIG_DIR / "default.yaml"
_LOCAL_CFG = _CONFIG_DIR / "local.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典，override 中的值覆盖 base。"""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _load_config() -> dict:
    """加载 default.yaml，若存在 local.yaml 则深度合并覆盖。"""
    with open(_DEFAULT_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if _LOCAL_CFG.exists():
        with open(_LOCAL_CFG, "r", encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, local)

    return cfg


def _resolve_path(rel_path: str) -> str:
    """将相对路径基于 PROJECT_ROOT 转为绝对路径。"""
    return str(PROJECT_ROOT / rel_path)


_cfg = _load_config()

# ============================================================
# YOLO 分割模型配置
# ============================================================

_yolo = _cfg.get("yolo", {})
YOLO_MODEL_PATH = _resolve_path(_yolo.get("model_path", "models/yolo/best.onnx"))
YOLO_CONF_THRESHOLD = float(_yolo.get("conf_threshold", 0.5))
YOLO_IOU_THRESHOLD = float(_yolo.get("iou_threshold", 0.7))
YOLO_DEVICE = str(_yolo.get("device", "cpu"))
YOLO_IMGSZ = int(_yolo.get("imgsz", 960))
YOLO_AUTO_INSTALL = bool(_yolo.get("auto_install", False))

if not YOLO_AUTO_INSTALL:
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# ============================================================
# 透视校正配置
# ============================================================

_rect = _cfg.get("rectifier", {})
RECTIFIER_EPSILON_RATIO = float(_rect.get("epsilon_ratio", 0.02))
RECTIFIER_USE_CONVEX_HULL = bool(_rect.get("use_convex_hull", True))
RECTIFIER_MORPH_SIZE = int(_rect.get("morph_size", 7))

# ============================================================
# 图像预处理配置
# ============================================================

_prep = _cfg.get("preprocessing", {})
PREPROCESSING_ENABLED = bool(_prep.get("enabled", True))
PREPROCESSING_MODE = str(_prep.get("mode", "auto"))  # "auto", "custom", "off"
PREPROCESSING_ENHANCE_CONTRAST = bool(_prep.get("enhance_contrast", True))
PREPROCESSING_DENOISE = bool(_prep.get("denoise", True))
PREPROCESSING_SHARPEN = bool(_prep.get("sharpen", False))
PREPROCESSING_ADJUST_BRIGHTNESS = bool(_prep.get("adjust_brightness", False))
PREPROCESSING_BINARIZE = bool(_prep.get("binarize", False))
PREPROCESSING_CONTRAST_CLIP_LIMIT = float(_prep.get("contrast_clip_limit", 2.0))
PREPROCESSING_DENOISE_STRENGTH = int(_prep.get("denoise_strength", 7))
PREPROCESSING_SHARPEN_STRENGTH = float(_prep.get("sharpen_strength", 0.5))
PREPROCESSING_TARGET_BRIGHTNESS = int(_prep.get("target_brightness", 130))

# ============================================================
# PaddleOCR 配置
# ============================================================

_ocr = _cfg.get("ocr", {})
OCR_MODEL_DIR = _resolve_path(_ocr.get("model_dir", "models/paddleocr"))
OCR_USE_ANGLE_CLS = bool(_ocr.get("use_angle_cls", True))
OCR_LANG = str(_ocr.get("lang", "ch"))
OCR_USE_GPU = bool(_ocr.get("use_gpu", False))
OCR_DET_DB_THRESH = float(_ocr.get("det_db_thresh", 0.2))
OCR_DET_DB_BOX_THRESH = float(_ocr.get("det_db_box_thresh", 0.4))
OCR_DET_DB_UNCLIP_RATIO = float(_ocr.get("det_db_unclip_ratio", 1.8))
ORIENTATION_THUMBNAIL_SIZE = int(_ocr.get("orientation_thumbnail_size", 640))

# ============================================================
# 输出配置
# ============================================================

_out = _cfg.get("output", {})
OUTPUT_DIR = _resolve_path(_out.get("dir", "output"))
SAVE_DEBUG_IMAGES = bool(_out.get("save_debug_images", True))
