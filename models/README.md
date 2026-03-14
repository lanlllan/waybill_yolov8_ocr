# 模型目录

## yolo/

存放 YOLO 分割模型（快递单检测 + 掩码）。

- **best.onnx**：主推理模型。可从项目 `export/export5/best.onnx` 复制到此。
- **yolov8n-seg-waybill.yaml**（可选）：类别等配置，可从 `export/export5/` 复制。

程序会优先使用本目录下的 `best.onnx`，若不存在则使用 `export/export5/best.onnx`。

## paddleocr/

PaddleOCR 模型可选存放位置。默认情况下 PaddleOCR 会将模型下载到用户目录；若需固定版本或离线部署，可将模型放置或链接到此目录并在代码中指定路径。
