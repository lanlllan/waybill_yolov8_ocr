# 快递单 OCR 提取系统

基于 YOLOv8-Seg 实例分割 + PaddleOCR 的快递单信息自动提取流水线。

**流程**：原始图片 → YOLO 分割定位 → 透视校正 → **图像预处理** → 方向矫正 → PaddleOCR 识别 → 结构化输出

## ✨ 主要特性

- ✅ **高精度分割**：基于 YOLOv8-Seg 的快递单定位
- ✅ **透视校正**：自动矫正任意角度和透视变形
- ✅ **智能预处理**：自动优化图像质量，提升 OCR 识别准确率（新增 🔥）
- ✅ **方向识别**：智能识别并矫正文字方向（0°/90°/180°/270°）
- ✅ **灵活配置**：YAML 配置文件，支持本地覆盖
- ✅ **完整输出**：标注图、过程图、文本、JSON 结构化结果

## 目录结构

```
waybill_ocr/                    ← 项目根目录
├── waybill_ocr/                # 核心代码包
│   ├── config.py               # 从 YAML 加载配置（支持 local.yaml 覆盖）
│   ├── segmentor.py            # YOLO 分割 + 标注图生成
│   ├── rectifier.py            # 透视校正（含 bbox 扩展）
│   ├── preprocessing.py        # 图像预处理（对比度增强、去噪、锐化）🆕
│   ├── ocr_engine.py           # PaddleOCR 封装 + 方向矫正
│   ├── ocr_engine_stub.py      # OCR 占位（未装 PaddleOCR 时降级使用）
│   └── pipeline.py             # 主流程串联 + 结果文件输出
├── config/
│   ├── default.yaml            # 默认配置（随仓库提交）
│   └── local.yaml              # 本地覆盖（不提交）
├── models/yolo/                # YOLO 模型（best.onnx，.gitignore 忽略）
├── data/input/                 # 待处理图片（.gitignore 忽略内容）
├── output/                     # 识别结果（按图片名分子文件夹）
├── models/paddleocr/           # PaddleOCR 模型缓存（首次运行自动下载）
├── tests/                      # 测试脚本
├── docs/
│   ├── DESIGN.md               # 完整方案设计文档
│   ├── TECHNIQUES.md           # 技术细节文档（论文参考）
│   ├── OCR_OPTIMIZATION.md     # OCR 优化指南（预处理详解）🆕
│   └── img/                    # 文档配图
├── requirements.txt            # Python 依赖
└── run_ocr.py                  # 入口脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> Python 3.8 用户需指定版本：`pip install paddleocr==2.7.3 paddlepaddle==2.6.2`

### 2. 准备模型

将 YOLOv8n-Seg 的 ONNX 模型放入 `models/yolo/best.onnx`。PaddleOCR 模型首次运行时自动下载。

### 3. 运行

```bash
# 处理 data/input 下所有图片
python run_ocr.py

# 指定图片
python run_ocr.py 图片1.jpg 图片2.jpg

# 保存汇总 JSON
python run_ocr.py --json

# 指定输出目录
python run_ocr.py -o my_output --json
```

未安装 PaddleOCR 时仍可跑通「分割 + 透视校正」，OCR 结果为空。

### 4. 输出结构

```
output/
├── 206-0/                      # 以图片名命名的子文件夹
│   ├── yolo_annotated.jpg      # YOLO 标注图（掩码+bbox+标签）
│   ├── 0_process.jpg           # 矫正过程可视化（各阶段中间结果拼接）
│   ├── 0_rectified.jpg         # 方向矫正后的最终图片
│   ├── 0_ocr_boxes.jpg         # OCR 检测框可视化（框+中文文字+置信度）
│   ├── 0_ocr.txt               # OCR 提取的纯文本
│   └── 0_result.json           # 完整结果（置信度、坐标、行信息）
└── results.json                # 汇总（--json 开启）
```

## 配置

配置文件位于 `config/default.yaml`，如需自定义，复制为 `config/local.yaml` 并修改：

```yaml
# YOLO 配置
yolo:
  device: "0"          # 切换到 GPU
  imgsz: 960           # ONNX 输入尺寸

# 图像预处理配置（新增）
preprocessing:
  enabled: true        # 启用预处理（推荐）
  mode: "auto"         # 自动模式，根据图像质量自适应

# OCR 配置
ocr:
  use_gpu: true
  det_db_thresh: 0.15  # 降低可检测更多模糊文字
```

`local.yaml` 只需写要覆盖的字段，其余沿用默认值。

### 图像预处理（提升 OCR 识别质量）🆕

系统内置图像预处理功能，可显著提升 OCR 识别准确率：

**自动模式（推荐）**：
```yaml
preprocessing:
  enabled: true
  mode: "auto"  # 自动评估图像质量并选择最优预处理方案
```

**自定义模式**：
```yaml
preprocessing:
  enabled: true
  mode: "custom"
  enhance_contrast: true    # 对比度增强（CLAHE）
  denoise: true             # 去噪处理
  sharpen: false            # 锐化（低质量图像可开启）
  adjust_brightness: false  # 亮度调整
  contrast_clip_limit: 2.0  # 对比度限制（1.0-4.0）
  denoise_strength: 7       # 去噪强度（3-15）
```

**适用场景**：
- ✅ 光照不均匀的图像
- ✅ 模糊或噪点多的图像
- ✅ 过暗或过亮的图像
- ✅ 对比度低的图像

详细说明请参考 [OCR 优化指南](docs/OCR_OPTIMIZATION.md)。

## 模型说明

| 用途 | 路径 | 说明 |
|------|------|------|
| YOLO 分割 | `models/yolo/best.onnx` | YOLOv8n-Seg，1 类 waybill，输入 960×960 |
| PaddleOCR | `models/paddleocr/` | PP-OCRv4，首次运行自动下载到项目内 |

## 运行测试

```bash
python tests/test_rectifier.py
```

## 文档

- [docs/DESIGN.md](docs/DESIGN.md) — 完整方案设计（各阶段说明、配置项、已解决问题）
- [docs/TECHNIQUES.md](docs/TECHNIQUES.md) — 技术细节（透视校正 + 方向矫正算法原理，含配图，面向论文写作）
- [docs/OCR_OPTIMIZATION.md](docs/OCR_OPTIMIZATION.md) — **OCR 优化指南**（图像预处理详解，提升识别质量）🆕
