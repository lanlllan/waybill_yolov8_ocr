# 快递单 OCR 提取系统

基于 YOLOv8-Seg 分割模型 + PaddleOCR 的快递单信息自动提取流水线。

## 目录结构

```
waybill_ocr/                    ← 项目根目录
├── waybill_ocr/                # 核心代码包
│   ├── __init__.py
│   ├── config.py               # 配置参数（模型路径、阈值等）
│   ├── segmentor.py            # YOLO 分割模块
│   ├── rectifier.py            # 透视校正模块
│   ├── ocr_engine.py           # PaddleOCR 封装 + 方向矫正
│   ├── ocr_engine_stub.py      # OCR 占位（未装 PaddleOCR 时使用）
│   └── pipeline.py             # 主流程串联
├── models/
│   ├── yolo/                   # YOLO 模型
│   │   ├── best.onnx           # YOLOv8n-Seg waybill
│   │   └── yolov8n-seg-waybill.yaml
│   └── paddleocr/              # PaddleOCR 本地缓存（可选）
├── data/
│   ├── input/                  # 待处理图片
│   └── samples/                # 样例/测试图片
├── output/                     # 识别结果、调试图、JSON
├── tests/
│   ├── test_rectifier.py       # 透视校正验证脚本
│   └── test_output/            # 测试输出图片
├── docs/
│   └── DESIGN.md               # 完整方案设计文档
├── config/                     # 运行时配置覆盖（可选）
├── logs/                       # 运行日志
├── requirements.txt            # Python 依赖
├── run_ocr.py                  # 入口脚本
└── README.md                   # 本说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行

在 **本目录**（`waybill_ocr/`）下执行：

```bash
# 处理 data/input 下所有图片，输出到 output/
python run_ocr.py

# 指定图片
python run_ocr.py 图片1.jpg 图片2.jpg

# 指定输出目录并保存 JSON
python run_ocr.py data/input/*.jpg -o output --json
```

未安装 PaddleOCR 时仍可跑通「分割 + 透视校正」，OCR 结果为空。

### 3. 运行测试

```bash
python tests/test_rectifier.py
```

## 模型说明

| 用途       | 路径                     | 说明 |
|------------|--------------------------|------|
| YOLO 分割  | `models/yolo/best.onnx`  | YOLOv8n-Seg，1 类 waybill |
| PaddleOCR  | `models/paddleocr/` 或系统默认 | 首次运行自动下载 |

## 设计文档

详见 `docs/DESIGN.md`。
