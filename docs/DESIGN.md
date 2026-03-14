# 快递单 OCR 提取 - 完整方案设计

## 整体流程

```
原始图片 → [阶段1] YOLO分割 → [阶段2] 透视校正 → [阶段3] 方向矫正 → [阶段4] PaddleOCR → 文字输出
```

## 项目目录结构

**`waybill_ocr/`** 为自包含的项目根目录，代码、模型、数据、输出全部在此。

```
waybill_ocr/                        ← 项目根目录
├── waybill_ocr/                    # 核心代码包
│   ├── __init__.py                 # 导出 WaybillSegmentor + rectifier 函数
│   ├── config.py                   # 所有配置参数，路径基于 PROJECT_ROOT
│   ├── segmentor.py                # 阶段1：YOLO 分割
│   ├── rectifier.py                # 阶段2：透视校正
│   ├── ocr_engine.py               # 阶段3+4：方向矫正 + PaddleOCR 识别
│   ├── ocr_engine_stub.py          # OCR 占位（未装 PaddleOCR 时自动降级）
│   └── pipeline.py                 # 阶段5：主流程串联
├── models/
│   ├── yolo/
│   │   ├── best.onnx               # YOLOv8n-Seg waybill（从 export5 转入）
│   │   └── yolov8n-seg-waybill.yaml
│   └── paddleocr/                  # PaddleOCR 本地缓存（可选，首次运行自动下载）
├── data/
│   ├── input/                      # 放入待处理图片
│   └── samples/                    # 样例/测试图片
├── output/                         # 识别结果、调试图、JSON
├── tests/
│   ├── test_rectifier.py           # 透视校正验证脚本（已通过）
│   ├── __init__.py
│   └── test_output/                # 测试输出图片（各角度+不规则掩码）
├── docs/
│   └── DESIGN.md                   # 本文档
├── config/                         # 运行时配置覆盖（可选，预留）
├── logs/                           # 运行日志（预留）
├── requirements.txt                # Python 依赖
├── run_ocr.py                      # 入口脚本
└── README.md                       # 项目说明
```

## 配置说明（config.py）

所有路径基于 `PROJECT_ROOT`（`waybill_ocr/` 目录），无外部依赖。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `YOLO_MODEL_PATH` | `models/yolo/best.onnx` | YOLO 分割模型路径 |
| `YOLO_CONF_THRESHOLD` | `0.5` | 检测置信度阈值 |
| `YOLO_IOU_THRESHOLD` | `0.7` | NMS IoU 阈值 |
| `YOLO_DEVICE` | `"cpu"` | 推理设备，`"cpu"` 或 `"0"`(GPU) |
| `OCR_USE_ANGLE_CLS` | `True` | PaddleOCR 启用方向分类 |
| `OCR_LANG` | `"ch"` | 中英文混合 |
| `OCR_USE_GPU` | `False` | OCR 使用 GPU |
| `OCR_DET_DB_THRESH` | `0.3` | 文本检测阈值（低值检测小字） |
| `ORIENTATION_THUMBNAIL_SIZE` | `640` | 方向矫正时缩略图最长边 |
| `OUTPUT_DIR` | `output/` | 输出目录 |
| `SAVE_DEBUG_IMAGES` | `True` | 保存中间步骤调试图 |

## 各阶段详细说明

---

### 阶段 1：YOLO 分割（segmentor.py）

**状态：已实现，未验证**

- **模型**：YOLOv8n-Seg，1 个类别 `waybill`，ONNX 格式
- **类**：`WaybillSegmentor(model_path, device, conf, iou)`
- **方法**：`segment(image: np.ndarray) -> list[dict]`
- **输入**：BGR 图像 (H, W, 3)
- **输出**：每个检测到的快递单返回：
  - `mask`：二值掩码 (H, W)，uint8，255 为前景
  - `bbox`：[x1, y1, x2, y2] numpy 数组
  - `confidence`：float 置信度
  - `class_id`：int 类别 ID
  - `class_name`：str 类别名
- **注意**：一张图中可能有多张快递单，逐个返回

---

### 阶段 2：透视校正（rectifier.py）

**状态：已完成，已验证 ✓**

#### 处理流程

1. **掩码清理 `clean_mask()`**
   - 闭运算（核 7×7，迭代 2 次）：填充内部孔洞
   - 开运算（核 7×7，迭代 1 次）：去除边缘突起
   - 最大连通区域过滤：去除离散碎片

2. **四角点检测 `find_quad_from_mask()`**
   - 凸包处理：`cv2.convexHull` 消除凹陷
   - 策略 A：`cv2.approxPolyDP` 逐步放大 epsilon（0.02~0.10）
     - 需通过退化检测（任意两点距离 > 5px）
     - 需通过凸性检查（`cv2.isContourConvex`）
   - 策略 B（兜底）：`cv2.minAreaRect` 最小外接旋转矩形

3. **角点排序 `order_points()`**
   - 按 y 坐标分为上下两组，每组内按 x 区分左右
   - 输出顺序：[左上, 右上, 右下, 左下]

4. **透视变换 `perspective_transform()`**
   - `cv2.getPerspectiveTransform` + `cv2.warpPerspective`

5. **一步调用 `rectify_from_mask(image, mask)`**
   - 返回 `rectified`（校正图）、`src_pts`、`target_size`、`cleaned_mask`

#### 验证结果

- 各角度（0°~180°）：角点误差 < 4px
- 不规则掩码（突起/缺口/碎片/孔洞/综合 5 种缺陷）：角点误差 < 17px

---

### 阶段 3：方向矫正（ocr_engine.py 内 `find_best_orientation`）

**状态：已实现，未验证**

#### 核心问题

透视校正后图像是正面矩形，但文字可能朝 0°/90°/180°/270° 四个方向之一。

#### 解决方案

1. PaddleOCR 的 `use_angle_cls=True` 自动处理 0°/180°
2. 对 90°/270°：四方向旋转择优
   - 竖向图（h > w）：只试 0°、180°
   - 横向图：试 0°、90°、180°、270°
   - 用缩略图（最长边 640px）做 OCR 加速
   - 按 `文字框数量 × 平均置信度` 打分，选最高分角度
3. 辅助函数：`_rotate_image(image, angle)`、`_resize_to_max(image, max_size)`

---

### 阶段 4：PaddleOCR 识别（ocr_engine.py 内 `WaybillOCR.recognize`）

**状态：已实现，未验证**

- **类**：`WaybillOCR(use_gpu, lang, use_angle_cls, det_db_thresh, thumbnail_size)`
- **方法**：`recognize(image: np.ndarray) -> dict`
- **返回**：
  - `full_text`：按行拼接的全文（`\n` 分隔）
  - `lines`：列表，每项含 `text`、`confidence`、`box`（4 角点）、`center`
  - `orientation`：选中的矫正角度

#### 结果排序

- 按 y 坐标量化分行（每 20px 视为同一行）
- 同行内按 x 坐标从左到右

#### 降级机制

未安装 PaddleOCR 时，`pipeline.py` 自动使用 `ocr_engine_stub.py`（返回空文本），分割+校正流程仍可跑通。

---

### 阶段 5：主流程（pipeline.py）

**状态：已实现，未验证**

- **类**：`WaybillPipeline(output_dir)`
- **方法**：
  - `process_image(image_path) -> list[dict]`：单张图片
  - `process_batch(image_paths) -> dict[str, list[dict]]`：批量
- **每个检测结果包含**：
  - `index`、`class`、`confidence`、`bbox`
  - `text`（全文）、`lines`（行列表）、`orientation`（矫正角度）
  - 异常时包含 `error` 字段
- **调试图保存**：`SAVE_DEBUG_IMAGES=True` 时写入 `output/debug_*_rectified.jpg`
- **命令行入口**：`python -m waybill_ocr.pipeline images... [-o output] [--json]`

---

### 阶段 6：生产部署

**状态：未实现**

#### 模型导出

| 模型 | 格式 | 场景 |
|------|------|------|
| YOLO | ONNX | CPU 部署（当前已使用） |
| YOLO | TensorRT | GPU 部署（需在目标 GPU 上导出） |
| PaddleOCR | ONNX | `use_onnx=True` |

#### API 服务

- FastAPI：`POST /api/ocr`、`POST /api/ocr/batch`
- 支持文件上传和 Base64

#### 容器化

- Dockerfile + docker-compose
- GPU 版本（nvidia-docker）和 CPU 版本

#### 性能预估

| 部署方式 | 单张耗时 |
|---------|---------|
| PyTorch + CPU | 3-8 秒 |
| ONNX + CPU | 1-3 秒 |
| TensorRT + GPU | 0.1-0.5 秒 |

---

## 开发进度总览

| 模块 | 文件 | 状态 | 测试 |
|------|------|------|------|
| YOLO 分割 | `segmentor.py` | 已实现 | 未验证 |
| 透视校正 | `rectifier.py` | 已实现 | **已验证 ✓** |
| 方向矫正 | `ocr_engine.py` | 已实现 | 未验证 |
| OCR 识别 | `ocr_engine.py` | 已实现 | 未验证 |
| 主流程 | `pipeline.py` | 已实现 | 未验证 |
| OCR 占位 | `ocr_engine_stub.py` | 已实现 | — |
| 配置管理 | `config.py` | 已实现 | — |
| 入口脚本 | `run_ocr.py` | 已实现 | — |
| 测试脚本 | `tests/test_rectifier.py` | 已完成 | ✓ |
| 生产部署 | — | 未实现 | — |

## 运行方式

```bash
# 在 waybill_ocr/ 目录下执行
pip install -r requirements.txt

# 处理 data/input 下所有图片
python run_ocr.py

# 指定图片
python run_ocr.py 图片1.jpg 图片2.jpg

# 保存 JSON
python run_ocr.py data/input/*.jpg -o output --json

# 运行透视校正测试
python tests/test_rectifier.py
```

## 下一步待办

1. **端到端验证**：用一张真实快递单图片跑通全流程（分割→校正→OCR）
2. **安装 PaddleOCR**：`pip install paddlepaddle paddleocr`，验证阶段 3+4
3. **生产部署**：FastAPI 封装、Docker 容器化
4. **性能优化**：GPU 推理、TensorRT 导出

## 来源说明

- 本项目从 `ultralytics-8.2.0` 仓库的 `ocr_export/` 目录迁移而来
- YOLO 模型从 `export/export5/best.onnx` 转入 `models/yolo/`
- 透视校正模块已在 `ocr_export/tests/` 中验证通过后迁移
