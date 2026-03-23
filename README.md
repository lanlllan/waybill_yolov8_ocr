# 快递单 OCR 提取系统

基于 YOLOv8-Seg 实例分割 + PaddleOCR 的快递单信息自动提取流水线。

**流程**：原始图片 → YOLO 分割定位 → 透视校正  → 方向矫正 → PaddleOCR 识别 → 结构化输出

## 目录结构

```
waybill_ocr/                    ← 项目根目录
├── waybill_ocr/                # 核心代码包
│   ├── config.py               # 从 YAML 加载配置（支持 local.yaml 覆盖）
│   ├── segmentor.py            # YOLO 分割 + 标注图生成
│   ├── rectifier.py            # 透视校正（含 bbox 扩展）
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

将 YOLOv8n-Seg 的 ONNX 模型放入 `models/yolo/best.onnx`）。若该文件不存在，程序会按 `config/default.yaml` 里的 `yolo.model_download_url` 自动下载到 `model_path` 所指路径。PaddleOCR 模型首次运行时下载到 `models/paddleocr/`（见 `ocr.model_dir`）。

### 3. 运行

```bash
# 处理 data/input 下所有图片
python run_ocr.py

# 指定图片
python run_ocr.py 图片1.jpg 图片2.jpg

# 汇总 results.json：默认会写入（由 output.save_results_json 控制）；--json 可显式强制写入
# 本次不写入汇总：python run_ocr.py --no-results-json

# 指定输出目录
python run_ocr.py -o my_output
```

未安装 PaddleOCR 时仍可跑通「分割 + 透视校正」，OCR 结果为空。

### 4. 输出结构

每张图子目录下 **始终** 写入 `{i}_ocr.txt`、`{i}_result.json`。`yolo_annotated.jpg`、`{i}_rectified.jpg`、`{i}_process.jpg`、`{i}_ocr_boxes.jpg` 仅在 `output.save_debug_images: true` 时生成（默认见 `config/default.yaml`）。

```
output/
├── 206-0/                      # 以图片名命名的子文件夹
│   ├── yolo_annotated.jpg      # 调试图：YOLO 标注（save_debug_images）
│   ├── 0_process.jpg           # 调试图：矫正过程拼接
│   ├── 0_rectified.jpg         # 调试图：方向矫正后图像
│   ├── 0_ocr_boxes.jpg         # 调试图：OCR 框 + 文字
│   ├── 0_ocr.txt               # OCR 纯文本
│   └── 0_result.json           # 完整结果（含 timing 等，见 timing.enabled）
└── results.json                # 批量汇总（默认写入；可用 save_results_json / --no-results-json）
```

## 配置

配置文件位于 `config/default.yaml`，如需自定义，复制为 `config/local.yaml` 并修改：

```yaml
yolo:
  device: "0"          # 切换到 GPU
  imgsz: 960           # ONNX 输入尺寸
  warmup: true         # 启动时空图预热，避免首张图承担 ONNX 冷启动
ocr:
  use_gpu: true
  det_db_thresh: 0.15  # 降低可检测更多模糊文字
output:
  save_debug_images: true   # 是否保存标注图、rectified、process、ocr_boxes
  save_results_json: true   # 是否在 output 根目录写 results.json
timing:
  enabled: false           # 控制台与 JSON 中的分阶段耗时
```

`local.yaml` 只需写要覆盖的字段，其余沿用默认值。

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
