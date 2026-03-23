# 快递单 OCR 提取 - 完整方案设计

## 整体流程

```
原始图片 → [阶段1] YOLO分割 → [阶段2] 透视校正 → [阶段3] 方向矫正 → [阶段4] PaddleOCR → 文字输出
                  ↓                    ↓                                        ↓
     yolo_annotated.jpg（调试图）  rectified 等（调试图）                    0_ocr.txt / 0_result.json（始终）
```

## 项目目录结构

```
waybill_ocr/                        ← 项目根目录
├── waybill_ocr/                    # 核心代码包
│   ├── __init__.py                 # 导出 WaybillSegmentor + rectifier 函数
│   ├── config.py                   # 从 YAML 加载配置，支持 local.yaml 覆盖
│   ├── segmentor.py                # 阶段1：YOLO 分割 + 标注图生成
│   ├── rectifier.py                # 阶段2：透视校正（含 bbox 扩展）
│   ├── ocr_engine.py               # 阶段3+4：方向矫正 + PaddleOCR 识别
│   ├── ocr_engine_stub.py          # OCR 占位（未装 PaddleOCR 时自动降级）
│   └── pipeline.py                 # 主流程串联 + 结果文件输出
├── models/
│   ├── yolo/
│   │   ├── best.onnx               # YOLOv8n-Seg waybill（ONNX，输入 960×960）
│   │   └── yolov8n-seg-waybill.yaml
│   └── paddleocr/                  # PaddleOCR 本地缓存（首次运行自动下载）
├── config/
│   ├── default.yaml                # 默认配置（随仓库提交）
│   └── local.yaml                  # 本地覆盖（.gitignore 已忽略）
├── data/
│   ├── input/                      # 放入待处理图片（.gitignore 忽略内容）
│   └── samples/                    # 样例/测试图片
├── output/                         # 识别结果（按图片名分子文件夹）
├── tests/
│   ├── test_rectifier.py           # 透视校正验证脚本（已通过）
│   ├── __init__.py
│   └── test_output/                # 测试输出图片
├── docs/
│   ├── DESIGN.md                   # 本文档（方案设计）
│   ├── TECHNIQUES.md               # 技术细节文档（论文参考）
│   └── img/                        # 文档配图
├── logs/                           # 运行日志（预留）
├── requirements.txt                # Python 依赖
├── run_ocr.py                      # 入口脚本
├── .gitignore
└── README.md                       # 项目说明
```

### 输出目录结构

每张输入图片生成独立的子文件夹：

```
output/
├── 206-0/                          # 图片名（去扩展名）
│   ├── yolo_annotated.jpg          # YOLO 标注图（掩码+bbox+标签+置信度）
│   ├── 0_process.jpg               # 矫正过程可视化（各阶段中间结果拼接）
│   ├── 0_rectified.jpg             # 方向矫正后的最终图片
│   ├── 0_ocr_boxes.jpg             # OCR 文本检测框可视化（框+中文文字+置信度）
│   ├── 0_ocr.txt                   # OCR 提取的纯文本（始终写入）
│   └── 0_result.json               # 完整结果（含置信度、坐标、行信息；timing 见配置）
└── results.json                    # 批量汇总（默认写入 output.save_results_json；CLI 可用 --no-results-json）
```

`yolo_annotated.jpg`、`{i}_rectified.jpg`、`{i}_process.jpg`、`{i}_ocr_boxes.jpg` 仅当 `output.save_debug_images: true` 时生成。多个快递单时按编号 `0_`、`1_`、`2_` 依次生成。

## 配置说明

### 配置加载机制

配置从 YAML 文件加载，支持两级覆盖：

1. `config/default.yaml` — 默认配置（随仓库提交）
2. `config/local.yaml` — 本地覆盖（.gitignore 忽略，不提交）

`local.yaml` 中只需写要覆盖的字段，其余自动使用 `default.yaml` 的值（深度合并）。

### 配置项一览

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| **YOLO 分割** | | |
| `yolo.model_path` | `models/yolo/best.onnx` | YOLO 分割模型路径 |
| `yolo.model_download_url` | （见 default.yaml） | 模型文件不存在时从此 URL 下载到 `model_path` |
| `yolo.conf_threshold` | `0.5` | 检测置信度阈值 |
| `yolo.iou_threshold` | `0.7` | NMS IoU 阈值 |
| `yolo.device` | `"cpu"` | 推理设备，`"cpu"` 或 `"0"`(GPU) |
| `yolo.imgsz` | `960` | ONNX 模型导出时的输入尺寸 |
| `yolo.auto_install` | `false` | 禁止 Ultralytics 自动安装缺失包 |
| `yolo.warmup` | `true` | 流水线启动时空图跑一次推理，摊薄 ONNX 冷启动 |
| **透视校正** | | |
| `rectifier.epsilon_ratio` | `0.02` | 轮廓近似精度系数 |
| `rectifier.use_convex_hull` | `true` | 是否使用凸包平滑 |
| `rectifier.morph_size` | `7` | 形态学核大小 |
| **PaddleOCR** | | |
| `ocr.use_angle_cls` | `true` | 启用方向分类（已弃用，实际以整图旋转对比为准） |
| `ocr.lang` | `"ch"` | 中英文混合 |
| `ocr.use_gpu` | `false` | OCR 使用 GPU |
| `ocr.det_db_thresh` | `0.2` | 文本区域检测阈值（低值检测模糊/遮挡文字） |
| `ocr.det_db_box_thresh` | `0.4` | 文本框置信度阈值（低值保留更多候选框） |
| `ocr.det_db_unclip_ratio` | `1.8` | 文本框扩展系数（大值合并断裂文字） |
| `ocr.orientation_thumbnail_size` | `640` | 方向矫正时缩略图最长边 |
| `ocr.model_dir` | `"models/paddleocr"` | PaddleOCR 模型本地缓存路径 |
| **输出** | | |
| `output.dir` | `"output"` | 输出目录 |
| `output.save_debug_images` | `false` | 是否保存 YOLO 标注图、rectified、process、ocr_boxes 等调试图 |
| `output.save_results_json` | `true` | 批量结束后是否写入根目录 `results.json` |
| **耗时** | | |
| `timing.enabled` | `true` | 控制台与 JSON 中输出分阶段耗时（整张 = 分解合计 + 其它；详见 `pipeline` 实现） |

### 本地覆盖示例 (`config/local.yaml`)

```yaml
yolo:
  device: "0"       # 切换到 GPU
ocr:
  use_gpu: true
  det_db_thresh: 0.15
```

## 各阶段详细说明

---

### 阶段 1：YOLO 分割（segmentor.py）

**状态：已完成，已验证 ✓**

- **模型**：YOLOv8n-Seg，1 个类别 `waybill`，ONNX 格式（输入 960×960）
- **类**：`WaybillSegmentor(model_path, device, conf, iou, imgsz)`
  - 加载时指定 `task="segment"` 解决 ONNX 元数据丢失问题
- **方法**：`segment(image) -> tuple[list[dict], np.ndarray | None]`
- **输入**：BGR 图像 (H, W, 3)
- **输出**：
  - `detections`：每个检测到的快递单返回：
    - `mask`：二值掩码 (H, W)，uint8，255 为前景
    - `bbox`：[x1, y1, x2, y2] numpy 数组
    - `confidence`：float 置信度
    - `class_id`：int 类别 ID
    - `class_name`：str 类别名
  - `annotated_image`：YOLO 绘制的标注图（含掩码+bbox+标签）
- **掩码生成**：优先使用 `r.masks.xy`（多边形坐标，已映射到原图坐标系）通过 `cv2.fillPoly` 生成精确掩码；仅在 `masks.xy` 不可用时回退到 `r.masks.data` resize
- **小面积过滤**：掩码面积小于图像总像素 0.5% 的检测结果自动丢弃，避免误检噪声

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

3. **bbox 扩展 `_expand_quad_with_bbox()`**（新增）
   - 当掩码四角点覆盖范围不到 YOLO bbox 的 75% 时，用 bbox 四角替代
   - 解决掩码不完整导致快递单内容被截断的问题

4. **角点排序 `order_points()`**
   - 按 y 坐标分为上下两组，每组内按 x 区分左右
   - 输出顺序：[左上, 右上, 右下, 左下]

5. **透视变换 `perspective_transform()`**
   - `cv2.getPerspectiveTransform` + `cv2.warpPerspective`

6. **一步调用 `rectify_from_mask(image, mask, bbox=None)`**
   - 接受可选的 `bbox` 参数用于扩展不完整掩码
   - 返回 `rectified`（校正图）、`src_pts`、`target_size`、`cleaned_mask`

#### 验证结果

- 各角度（0°~180°）：角点误差 < 4px
- 不规则掩码（突起/缺口/碎片/孔洞/综合 5 种缺陷）：角点误差 < 17px
- 真实快递单图片：端到端验证通过

---

### 阶段 3：方向矫正（ocr_engine.py 内 `find_best_orientation`）

**状态：已完成，已验证 ✓**

#### 核心问题

透视校正后图像是正面矩形，但文字可能朝 0°/90°/180°/270° 四个方向之一。

#### 解决方案：整图旋转对比

对所有方向统一采用**整图旋转 + OCR 评分对比**，不依赖 PaddleOCR 的 `angle_cls` 逐行分类器（`cls=False`），避免模糊文字被逐行误判。

1. **0°/180° 对比**：始终执行，解决倒置拍摄问题
2. **90°/270° 对比**：仅在横向图（`h < w × 0.8`）时额外执行，解决竖版快递单横拍问题
3. **评分函数**：`score = Σ(conf_i × len(text_i))`（仅计 conf ≥ 0.7 的检测框）
4. **0° 优先门槛**：其他方向得分需超出当前最佳的 **150%** 才会替换（高门槛防止噪声误判）
5. **最终识别也用 `cls=False`**：方向已由整图对比确定，不需要 `angle_cls` 再逐行翻转
6. 辅助函数：`_rotate_image(image, angle)`、`_resize_to_max(image, max_size)`、`_score_orientation(image)`

---

### 阶段 4：PaddleOCR 识别（ocr_engine.py 内 `WaybillOCR.recognize`）

**状态：已完成，已验证 ✓**

- **类**：`WaybillOCR(use_gpu, lang, use_angle_cls, det_db_thresh, det_db_box_thresh, det_db_unclip_ratio, thumbnail_size, model_dir)`
- **方法**：`recognize(image: np.ndarray) -> dict`
- **返回**：
  - `full_text`：按行拼接的全文（`\n` 分隔）
  - `lines`：列表，每项含 `text`、`confidence`、`box`（4 角点）、`center`
  - `orientation`：选中的矫正角度
  - `rotated_image`：方向矫正后的图像（供 pipeline 用于保存和可视化）

#### 结果排序

- 按 y 坐标量化分行（每 20px 视为同一行）
- 同行内按 x 坐标从左到右

#### 降级机制

未安装 PaddleOCR 时，`pipeline.py` 自动使用 `ocr_engine_stub.py`（返回空文本），分割+校正流程仍可跑通。

---

### 阶段 5：主流程（pipeline.py）

**状态：已完成，已验证 ✓**

- **类**：`WaybillPipeline(output_dir)`
- **方法**：
  - `process_image(image_path) -> list[dict]`：单张图片
  - `process_batch(image_paths) -> dict[str, list[dict]]`：批量
- **每个检测结果包含**：
  - `index`、`class`、`confidence`、`bbox`
  - `text`（全文）、`lines`（行列表）、`orientation`（矫正角度）
  - 异常时包含 `error` 字段
- **输出文件**：每张图片一个子文件夹；**始终**写入 `{i}_ocr.txt`、`{i}_result.json`。`yolo_annotated.jpg`、`{i}_process.jpg`、`{i}_rectified.jpg`、`{i}_ocr_boxes.jpg` 仅当 `output.save_debug_images: true` 时写入。
- **命令行入口**：`python run_ocr.py [images...] [-o output] [--json] [--no-results-json]`（`--json` 与配置任一为真即写 `results.json`；`--no-results-json` 可覆盖配置关闭汇总）

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
| YOLO 分割 | `segmentor.py` | 已完成 | **已验证 ✓** |
| 透视校正 | `rectifier.py` | 已完成 | **已验证 ✓** |
| bbox 扩展 | `rectifier.py` | 已完成 | **已验证 ✓** |
| 方向矫正 | `ocr_engine.py` | 已完成 | **已验证 ✓** |
| OCR 识别 | `ocr_engine.py` | 已完成 | **已验证 ✓** |
| 主流程 | `pipeline.py` | 已完成 | **已验证 ✓** |
| 结果输出 | `pipeline.py` | 已完成 | **已验证 ✓** |
| OCR 占位 | `ocr_engine_stub.py` | 已完成 | — |
| 配置管理 | `config.py` + YAML | 已完成 | **已验证 ✓** |
| 入口脚本 | `run_ocr.py` | 已完成 | **已验证 ✓** |
| 测试脚本 | `tests/test_rectifier.py` | 已完成 | ✓ |
| 生产部署 | — | 未实现 | — |

## 已解决的关键问题

1. **ONNX 输入尺寸不匹配**：模型导出时用 960×960，需在 predict 时指定 `imgsz=960`
2. **ONNX task 元数据丢失**：加载时需显式指定 `task="segment"`
3. **小面积误检**：segmentor 中增加掩码面积过滤（< 0.5% 总像素则丢弃）
4. **掩码不完整导致截断**：当掩码四角点覆盖范围不到 bbox 的 75% 时，用 bbox 扩展
5. **掩码宽高比失真**：原先直接 resize `r.masks.data`（低分辨率正方形）到原图非正方形尺寸导致形变，改用 `r.masks.xy` 多边形 + `cv2.fillPoly` 生成精确掩码
6. **方向矫正误判**：弃用 PaddleOCR 逐行 `angle_cls`（对模糊文字不可靠），改为整图 0°/90°/180°/270° 四方向 OCR 评分对比，设 1.5× 门槛防止噪声误判
7. **被遮挡文字未检测**：调低 `det_db_thresh` 和 `det_db_box_thresh`，增大 `det_db_unclip_ratio`
8. **PaddleOCR 3.x 不兼容 Python 3.8**：降级到 paddleocr 2.7.3 + paddlepaddle 2.6.2
9. **PaddleOCR 模型路径不可控**：通过 `ocr.model_dir` 配置项指定本地缓存路径
10. **OCR 标注中文乱码**：用 Pillow + 系统字体 `msyh.ttc` 替代 `cv2.putText` 绘制中文
11. **方向矫正后图像未正确输出**：`recognize()` 返回 `rotated_image`，pipeline 使用矫正后图像保存和可视化

## 运行方式

```bash
# 安装依赖（Python 3.8 需用 paddleocr==2.7.3 paddlepaddle==2.6.2）
pip install -r requirements.txt

# 处理 data/input 下所有图片
python run_ocr.py

# 指定图片
python run_ocr.py 图片1.jpg 图片2.jpg

# 汇总 results.json（默认开启；可用 output.save_results_json 或 --no-results-json 控制）
python run_ocr.py --no-results-json

# 指定输出目录
python run_ocr.py -o my_output

# 运行透视校正测试
python tests/test_rectifier.py
```

## 依赖版本

| 包 | 版本要求 | 说明 |
|---|---|---|
| opencv-python | >=4.5.0 | 图像处理 |
| numpy | >=1.20.0 | 数组计算 |
| ultralytics | >=8.0.0 | YOLOv8 推理 |
| paddlepaddle | 2.6.2 | PaddleOCR 后端（Python 3.8 兼容） |
| paddleocr | 2.7.3 | OCR 识别（Python 3.8 兼容） |
| pyyaml | >=6.0 | YAML 配置加载 |

## 下一步待办

1. **多图验证**：用更多真实快递单图片测试，收集边界场景
2. **生产部署**：FastAPI 封装、Docker 容器化
3. **性能优化**：GPU 推理、TensorRT 导出
4. **Python 升级**：升级到 3.9+ 后可使用 paddleocr 3.x（模型更新、精度更高）

## 来源说明

- 本项目从 `ultralytics-8.2.0` 仓库的 `ocr_export/` 目录迁移而来
- YOLO ONNX 放在 `models/yolo/best.onnx`（可由 `model_download_url` 自动拉取）
- 透视校正模块已在 `ocr_export/tests/` 中验证通过后迁移
