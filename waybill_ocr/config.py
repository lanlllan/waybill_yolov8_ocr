"""
集中管理所有可配置参数。

加载顺序（后者覆盖前者）：
    1. config/default.yaml   — 默认配置（随仓库提交）
    2. config/local.yaml     — 本地覆盖（不提交，.gitignore 已忽略）

路径字段（如 model_path、dir）自动基于 PROJECT_ROOT 解析为绝对路径。
若配置了 yolo.model_download_url 且模型文件不存在，会在导入本模块时下载到 model_path。
"""

from __future__ import annotations

import os
import urllib.request
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


def _float_list(value, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    try:
        return [float(x) for x in value]
    except TypeError:
        return default


def _int_list(value, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    try:
        return [int(x) for x in value]
    except TypeError:
        return default


def _download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """将 url 下载到 dest（先写 .part 再替换，支持大文件）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "waybill-yolov8-ocr/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _ensure_yolo_model(model_path: str, download_url: str | None) -> None:
    """若模型文件不存在且配置了下载地址，则下载到 model_path。"""
    path = Path(model_path)
    if path.is_file():
        return
    url = (download_url or "").strip()
    if not url:
        return
    print(f"未找到 YOLO 模型: {path}\n正在下载: {url}")
    _download_file(url, path)
    print(f"已保存: {path}")


_cfg = _load_config()

# ============================================================
# YOLO 分割模型配置
# ============================================================

_yolo = _cfg.get("yolo", {})
YOLO_MODEL_PATH = _resolve_path(_yolo.get("model_path", "models/yolo/best.onnx"))
YOLO_MODEL_DOWNLOAD_URL = _yolo.get("model_download_url") or None
if isinstance(YOLO_MODEL_DOWNLOAD_URL, str):
    YOLO_MODEL_DOWNLOAD_URL = YOLO_MODEL_DOWNLOAD_URL.strip() or None
_ensure_yolo_model(YOLO_MODEL_PATH, YOLO_MODEL_DOWNLOAD_URL)
YOLO_CONF_THRESHOLD = float(_yolo.get("conf_threshold", 0.5))
YOLO_IOU_THRESHOLD = float(_yolo.get("iou_threshold", 0.7))
YOLO_DEVICE = str(_yolo.get("device", "cpu"))
YOLO_IMGSZ = int(_yolo.get("imgsz", 960))
YOLO_WARMUP = bool(_yolo.get("warmup", True))
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
RECTIFIER_MORPH_CLOSE_ITERATIONS = int(_rect.get("morph_close_iterations", 2))
RECTIFIER_MORPH_OPEN_ITERATIONS = int(_rect.get("morph_open_iterations", 1))
RECTIFIER_APPROX_RATIO_MULTIPLIERS = _float_list(
    _rect.get("approx_ratio_multipliers"),
    [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
)
RECTIFIER_BBOX_COVERAGE_THRESHOLD = float(_rect.get("bbox_coverage_threshold", 0.75))

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
ORIENTATION_CONF_THRESHOLD = float(_ocr.get("orientation_conf_threshold", 0.7))
ORIENTATION_EARLY_ACCEPT_0 = bool(_ocr.get("orientation_early_accept_0", False))
ORIENTATION_HIGH_CONF_THRESHOLD = float(_ocr.get("orientation_high_conf_threshold", 0.9))
ORIENTATION_ACCEPT_MIN_LINES = int(_ocr.get("orientation_accept_min_lines", 8))
ORIENTATION_ACCEPT_MIN_SCORE = float(_ocr.get("orientation_accept_min_score", 80.0))
ORIENTATION_PREFER_0_RATIO = float(_ocr.get("orientation_prefer_0_ratio", 1.5))
ORIENTATION_CANDIDATE_ANGLES = _int_list(
    _ocr.get("orientation_candidate_angles"),
    [180, 90, 270],
)
OCR_LINE_SORT_Y_BUCKET = int(_ocr.get("line_sort_y_bucket", 20))

# ============================================================
# 输出配置
# ============================================================

_out = _cfg.get("output", {})
OUTPUT_DIR = _resolve_path(_out.get("dir", "output"))
SAVE_DEBUG_IMAGES = bool(_out.get("save_debug_images", True))
SAVE_RESULTS_JSON = bool(_out.get("save_results_json", True))

# ============================================================
# 耗时统计
# ============================================================

_timing = _cfg.get("timing", {})
TIMING_ENABLED = bool(_timing.get("enabled", False))
