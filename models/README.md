# 模型目录

## yolo/

存放 YOLO 分割模型（快递单检测 + 掩码）。

- **best.onnx**：主推理模型。可手动放入本目录；若缺失，程序会在加载配置时按 `config/default.yaml` 中的 `yolo.model_download_url` 自动下载到 `yolo.model_path`（默认即本目录下的 `best.onnx`）。
- **yolov8n-seg-waybill.yaml**（可选）：类别与导出元数据说明，随仓库提供；推理以 `best.onnx` 为准。

## paddleocr/

PaddleOCR 权重默认缓存在 `models/paddleocr/`（由配置项 `ocr.model_dir` 指定）。首次运行会自动下载；离线环境可预先放入该目录。
