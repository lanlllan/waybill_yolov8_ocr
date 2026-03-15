"""
快递单 OCR 入口脚本。

    python run_ocr.py [图片路径 ...]
    python run_ocr.py                  # 默认处理 data/input 下图片

输出结构：
    output/
    ├── <图片名>/
    │   ├── 0_rectified.jpg    # 校正图
    │   ├── 0_ocr.txt          # OCR 文本
    │   └── 0_result.json      # 完整结果
    └── results.json           # 汇总（--json 开启）
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from waybill_ocr.pipeline import WaybillPipeline, _make_serializable


def main():
    parser = argparse.ArgumentParser(description="快递单 OCR 提取")
    parser.add_argument(
        "images",
        nargs="*",
        help="图片路径；不传则使用 data/input 下图片",
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(PROJECT_ROOT, "output"),
        help="输出目录，默认 output/",
    )
    parser.add_argument("--json", "-j", action="store_true", help="保存汇总 results.json")
    args = parser.parse_args()

    if not args.images:
        input_dir = os.path.join(PROJECT_ROOT, "data", "input")
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            args.images.extend(glob.glob(os.path.join(input_dir, ext)))
        args.images.sort()
        if not args.images:
            print(f"未指定图片且 {input_dir} 下无图片，请传入图片路径或放入 data/input 目录")
            return

    os.makedirs(args.output, exist_ok=True)
    pipeline = WaybillPipeline(output_dir=args.output)
    results = pipeline.process_batch(args.images)

    for path, items in results.items():
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== {path} → output/{stem}/ ===")
        for item in items:
            if "error" in item:
                print(f"  错误: {item['error']}")
                continue
            print(f"  快递单 #{item['index']} "
                  f"(置信度: {item['confidence']:.2f}, "
                  f"方向: {item['orientation']}°)")
            print(f"  {item['text']}")

    if args.json:
        json_path = os.path.join(args.output, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(results), f,
                      ensure_ascii=False, indent=2)
        print(f"\n汇总已保存: {json_path}")


if __name__ == "__main__":
    main()
