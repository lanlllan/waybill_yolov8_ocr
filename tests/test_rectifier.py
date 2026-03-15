"""
透视校正模块的独立验证脚本。

测试 1（合成测试）：生成已知透视变形的矩形图，验证各角度校正精度
测试 2（不规则掩码）：模拟 YOLO 分割输出的各种缺陷，验证鲁棒性
测试 3（真实图片）：用实际图片 + 自动检测掩码测试

用法:
    # 在 waybill_ocr 项目根目录下运行
    python tests/test_rectifier.py                    # 运行测试 1 + 2
    python tests/test_rectifier.py --image 图片路径     # 额外运行测试 3

输出:
    所有结果图片保存在 tests/test_output/ 目录
"""

import argparse
import os
import sys

import cv2
import numpy as np

# 添加项目根目录到 Python 路径，使 waybill_ocr 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from waybill_ocr.rectifier import (
    clean_mask,
    draw_quad_on_image,
    find_quad_from_mask,
    order_points,
    rectify_from_mask,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 辅助函数：生成测试数据
# ============================================================

def create_synthetic_waybill(width=400, height=600) -> np.ndarray:
    """创建一个带文字和线条的模拟快递单图像。"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240

    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)
    cv2.line(img, (10, 80), (width - 10, 80), (0, 0, 0), 1)
    cv2.line(img, (10, 200), (width - 10, 200), (0, 0, 0), 1)
    cv2.line(img, (10, 350), (width - 10, 350), (0, 0, 0), 1)

    cv2.putText(img, "EXPRESS WAYBILL", (50, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(img, "From: Sender Name", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Addr: 123 Street", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Phone: 138-0000-0000", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "To: Receiver Name", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Addr: 456 Avenue", (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "NO: SF1234567890", (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)

    return img


def apply_perspective_distortion(img: np.ndarray, angle_deg: float = 30,
                                 canvas_size: int = 900) -> tuple:
    """
    对图像施加透视变形 + 旋转，模拟真实拍摄场景。

    Args:
        img: 原始快递单图像
        angle_deg: 旋转角度
        canvas_size: 画布大小

    Returns:
        (distorted_canvas, mask_canvas, src_quad_on_canvas)
    """
    h, w = img.shape[:2]
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    squeeze = 0.15
    tilt = 0.08
    dst_pts = np.array([
        [w * squeeze, h * tilt],
        [w * (1 - squeeze), 0],
        [w * 0.95, h * 0.95],
        [w * 0.05, h],
    ], dtype=np.float32)

    M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M_persp, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(128, 128, 128))

    warped_corners = cv2.perspectiveTransform(
        src_pts.reshape(1, -1, 2), M_persp
    ).reshape(4, 2)

    canvas = np.full((canvas_size, canvas_size, 3), 128, dtype=np.uint8)
    cx, cy = canvas_size // 2, canvas_size // 2
    ox, oy = w // 2, h // 2
    offset_x, offset_y = cx - ox, cy - oy

    y_start = max(0, offset_y)
    y_end = min(canvas_size, offset_y + h)
    x_start = max(0, offset_x)
    x_end = min(canvas_size, offset_x + w)
    src_y_start = max(0, -offset_y)
    src_x_start = max(0, -offset_x)
    canvas[y_start:y_end, x_start:x_end] = warped[
        src_y_start:src_y_start + (y_end - y_start),
        src_x_start:src_x_start + (x_end - x_start)
    ]

    corners_on_canvas = warped_corners + np.array([offset_x, offset_y])

    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    canvas_rotated = cv2.warpAffine(canvas, M_rot, (canvas_size, canvas_size),
                                    borderValue=(128, 128, 128))

    ones = np.ones((4, 1))
    corners_h = np.hstack([corners_on_canvas, ones])
    rotated_corners = (M_rot @ corners_h.T).T

    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    pts_int = rotated_corners.astype(np.int32)
    cv2.fillPoly(mask, [pts_int], 255)

    return canvas_rotated, mask, rotated_corners.astype(np.float32)


def add_mask_noise(mask: np.ndarray, seed: int = 42) -> dict:
    """
    给干净掩码添加各种缺陷，模拟 YOLO 分割输出的不完美情况。

    Returns:
        dict: {缺陷名称: 带缺陷的掩码}
            - bumpy:      边缘突起（轮廓外围随机凸起）
            - notched:    边缘缺口（随机在边缘挖去区域）
            - fragmented: 离散碎片（掩码外有独立噪点区域）
            - holed:      内部孔洞（掩码内部有洞）
            - combined:   全部叠加
    """
    rng = np.random.RandomState(seed)
    h, w = mask.shape[:2]
    results = {}

    # 1. 边缘突起
    bumpy = mask.copy()
    contours, _ = cv2.findContours(bumpy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        for _ in range(8):
            idx = rng.randint(0, len(cnt))
            pt = cnt[idx][0]
            cv2.circle(bumpy, tuple(pt), rng.randint(10, 30), 255, -1)
    results["bumpy"] = bumpy

    # 2. 边缘缺口
    notched = mask.copy()
    contours, _ = cv2.findContours(notched, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        for _ in range(6):
            idx = rng.randint(0, len(cnt))
            pt = cnt[idx][0]
            cv2.circle(notched, tuple(pt), rng.randint(10, 25), 0, -1)
    results["notched"] = notched

    # 3. 离散碎片
    fragmented = mask.copy()
    for _ in range(5):
        cx = rng.randint(50, w - 50)
        cy = rng.randint(50, h - 50)
        if mask[cy, cx] == 0:
            cv2.circle(fragmented, (cx, cy), rng.randint(5, 20), 255, -1)
    results["fragmented"] = fragmented

    # 4. 内部孔洞
    holed = mask.copy()
    contours, _ = cv2.findContours(holed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx_center = int(M["m10"] / M["m00"])
            cy_center = int(M["m01"] / M["m00"])
            for _ in range(4):
                ox = rng.randint(-50, 50)
                oy = rng.randint(-50, 50)
                cv2.circle(holed, (cx_center + ox, cy_center + oy),
                           rng.randint(8, 20), 0, -1)
    results["holed"] = holed

    # 5. 综合
    combined = mask.copy()
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        for _ in range(5):
            idx = rng.randint(0, len(cnt))
            pt = cnt[idx][0]
            cv2.circle(combined, tuple(pt), rng.randint(10, 25), 255, -1)
        for _ in range(4):
            idx = rng.randint(0, len(cnt))
            pt = cnt[idx][0]
            cv2.circle(combined, tuple(pt), rng.randint(8, 20), 0, -1)
    for _ in range(3):
        cx = rng.randint(50, w - 50)
        cy = rng.randint(50, h - 50)
        if mask[cy, cx] == 0:
            cv2.circle(combined, (cx, cy), rng.randint(5, 15), 255, -1)
    results["combined"] = combined

    return results


def add_edge_irregular_noise(mask: np.ndarray, seed: int = 99) -> dict:
    """
    给干净掩码添加边缘整体不规则缺陷，模拟 YOLO 分割的典型输出。

    与 add_mask_noise 中的局部缺陷不同，这些缺陷影响的是整条边缘。

    Returns:
        dict: {缺陷名称: 带缺陷的掩码}
            - jagged:   锯齿边缘（模拟低分辨率掩码上采样）
            - wavy:     波浪边缘（模拟分割边界不确定性）
            - eroded:   整体侵蚀（掩码向内均匀收缩）
            - blunted:  角点钝化（四角被磨圆成弧形）
    """
    rng = np.random.RandomState(seed)
    h, w = mask.shape[:2]
    results = {}

    # 1. 锯齿边缘：缩小再放大模拟低分辨率上采样
    scale = 8
    small = cv2.resize(mask, (w // scale, h // scale),
                       interpolation=cv2.INTER_NEAREST)
    jagged = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    results["jagged"] = jagged

    # 2. 波浪边缘：对轮廓点施加法向正弦扰动
    wavy = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if contours:
        cnt = max(contours, key=cv2.contourArea).copy()
        n_pts = len(cnt)
        amplitude = 12
        freq = 15
        for i in range(n_pts):
            offset = amplitude * np.sin(2 * np.pi * freq * i / n_pts)
            prev_idx = (i - 1) % n_pts
            next_idx = (i + 1) % n_pts
            dx = float(cnt[next_idx][0][0] - cnt[prev_idx][0][0])
            dy = float(cnt[next_idx][0][1] - cnt[prev_idx][0][1])
            length = max(np.sqrt(dx * dx + dy * dy), 1e-6)
            nx, ny = -dy / length, dx / length
            cnt[i][0][0] = int(cnt[i][0][0] + nx * offset)
            cnt[i][0][1] = int(cnt[i][0][1] + ny * offset)
        cv2.drawContours(wavy, [cnt], -1, 255, -1)
    results["wavy"] = wavy

    # 3. 整体侵蚀：均匀内缩 15px
    erode_px = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (erode_px * 2 + 1, erode_px * 2 + 1))
    eroded = cv2.erode(mask, kernel, iterations=1)
    results["eroded"] = eroded

    # 4. 角点钝化：对掩码做大半径高斯模糊后重新二值化
    blur_k = 51
    blurred = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    _, blunted = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    results["blunted"] = blunted

    return results


# ============================================================
# 测试 1：合成透视变形校正（各角度）
# ============================================================

def test_synthetic():
    """对 0°~180° 的合成透视变形图测试校正精度。"""
    ensure_output_dir()
    print("=" * 60)
    print("测试 1：合成透视变形校正")
    print("=" * 60)

    waybill = create_synthetic_waybill(400, 600)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_original_waybill.jpg"), waybill)
    print(f"[1/6] 生成模拟快递单: {waybill.shape[1]}x{waybill.shape[0]}")

    test_angles = [0, 30, 45, 90, 135, 180]
    for angle in test_angles:
        print(f"\n--- 测试旋转角度: {angle} ---")

        distorted, mask, true_corners = apply_perspective_distortion(
            waybill, angle_deg=angle
        )
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"02_distorted_{angle}deg.jpg"), distorted)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"03_mask_{angle}deg.jpg"), mask)
        print(f"[2/6] 施加透视变形 + 旋转 {angle}")

        detected_pts = find_quad_from_mask(mask)
        print(f"[3/6] 检测到四角点:")
        for label, pt in zip(["TL", "TR", "BR", "BL"], detected_pts):
            print(f"       {label}: ({pt[0]:.1f}, {pt[1]:.1f})")

        vis = draw_quad_on_image(distorted, detected_pts)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"04_detected_quad_{angle}deg.jpg"), vis)
        print(f"[4/6] 已保存角点可视化")

        result = rectify_from_mask(distorted, mask)
        rectified = result["rectified"]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"05_rectified_{angle}deg.jpg"), rectified)
        print(f"[5/6] 校正后尺寸: {rectified.shape[1]}x{rectified.shape[0]}")

        true_ordered = order_points(true_corners)
        corner_errors = np.linalg.norm(detected_pts - true_ordered, axis=1)
        print(f"[6/6] 角点误差 - 平均: {corner_errors.mean():.1f}px, "
              f"最大: {corner_errors.max():.1f}px")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


# ============================================================
# 测试 2：不规则掩码校正
# ============================================================

def test_irregular_masks():
    """测试不规则掩码（模拟 YOLO 分割）的校正鲁棒性。"""
    ensure_output_dir()
    print("\n" + "=" * 60)
    print("测试 2：不规则掩码校正（模拟 YOLO 分割）")
    print("=" * 60)

    waybill = create_synthetic_waybill(400, 600)
    distorted, clean_mask_img, true_corners = apply_perspective_distortion(
        waybill, angle_deg=20
    )

    noisy_masks = add_mask_noise(clean_mask_img)

    for name, noisy_mask in noisy_masks.items():
        print(f"\n--- 缺陷类型: {name} ---")

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"irreg_{name}_01_noisy.jpg"), noisy_mask)

        cleaned = clean_mask(noisy_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"irreg_{name}_02_cleaned.jpg"), cleaned)

        intersection = cv2.bitwise_and(clean_mask_img, cleaned)
        union = cv2.bitwise_or(clean_mask_img, cleaned)
        iou = cv2.countNonZero(intersection) / max(cv2.countNonZero(union), 1)
        print(f"  掩码清理 IoU: {iou:.3f}")

        try:
            result = rectify_from_mask(distorted, noisy_mask)
            rectified = result["rectified"]
            src_pts = result["src_pts"]

            vis = draw_quad_on_image(distorted, src_pts)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"irreg_{name}_03_quad.jpg"), vis)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"irreg_{name}_04_rect.jpg"), rectified)

            true_ordered = order_points(true_corners)
            errors = np.linalg.norm(src_pts - true_ordered, axis=1)
            print(f"  校正后: {rectified.shape[1]}x{rectified.shape[0]}, "
                  f"误差: avg={errors.mean():.1f}px, max={errors.max():.1f}px -> OK")
        except ValueError as e:
            print(f"  校正失败: {e}")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


# ============================================================
# 测试 2b：边缘整体不规则掩码校正
# ============================================================

def test_edge_irregular_masks():
    """测试边缘整体不规则掩码（锯齿/波浪/侵蚀/钝化）的校正鲁棒性。"""
    ensure_output_dir()
    print("\n" + "=" * 60)
    print("测试 2b：边缘整体不规则掩码校正")
    print("=" * 60)

    waybill = create_synthetic_waybill(400, 600)
    distorted, clean_mask_img, true_corners = apply_perspective_distortion(
        waybill, angle_deg=20
    )

    noisy_masks = add_edge_irregular_noise(clean_mask_img)

    for name, noisy_mask in noisy_masks.items():
        print(f"\n--- 缺陷类型: {name} ---")

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"edge_{name}_01_noisy.jpg"), noisy_mask)

        cleaned = clean_mask(noisy_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"edge_{name}_02_cleaned.jpg"), cleaned)

        intersection = cv2.bitwise_and(clean_mask_img, cleaned)
        union = cv2.bitwise_or(clean_mask_img, cleaned)
        iou = cv2.countNonZero(intersection) / max(cv2.countNonZero(union), 1)
        print(f"  掩码清理 IoU: {iou:.3f}")

        try:
            result = rectify_from_mask(distorted, noisy_mask)
            rectified = result["rectified"]
            src_pts = result["src_pts"]

            vis = draw_quad_on_image(distorted, src_pts)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"edge_{name}_03_quad.jpg"), vis)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"edge_{name}_04_rect.jpg"), rectified)

            true_ordered = order_points(true_corners)
            errors = np.linalg.norm(src_pts - true_ordered, axis=1)
            print(f"  校正后: {rectified.shape[1]}x{rectified.shape[0]}, "
                  f"误差: avg={errors.mean():.1f}px, max={errors.max():.1f}px -> OK")
        except ValueError as e:
            print(f"  校正失败: {e}")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


# ============================================================
# 测试 3：真实图片（可选）
# ============================================================

def test_with_real_image(image_path: str):
    """用真实图片测试，自动检测快递单区域。"""
    ensure_output_dir()
    print("\n" + "=" * 60)
    print("测试 3：真实图片透视校正")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"错误: 图片不存在 - {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片 - {image_path}")
        return

    print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")
    print("尝试自动检测快递单区域（基于颜色）...")

    mask = auto_detect_waybill_mask(image)
    if mask is not None and cv2.countNonZero(mask) > 1000:
        print(f"检测成功，掩码面积: {cv2.countNonZero(mask)} px")
    else:
        print("自动检测失败，使用中心 60% 区域")
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mx, my = int(w * 0.2), int(h * 0.2)
        mask[my:h - my, mx:w - mx] = 255

    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_01_input.jpg"), image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_02_mask.jpg"), mask)

    try:
        result = rectify_from_mask(image, mask)
        vis = draw_quad_on_image(image, result["src_pts"])
        cv2.imwrite(os.path.join(OUTPUT_DIR, "real_03_quad.jpg"), vis)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "real_04_rectified.jpg"), result["rectified"])
        print(f"校正后: {result['rectified'].shape[1]}x{result['rectified'].shape[0]}")
        print(f"结果保存到: {OUTPUT_DIR}")
    except ValueError as e:
        print(f"校正失败: {e}")


def auto_detect_waybill_mask(image: np.ndarray) -> np.ndarray:
    """基于颜色和轮廓的简单快递单区域检测（不依赖 YOLO）。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < image.shape[0] * image.shape[1] * 0.05:
        return None

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="透视校正模块验证")
    parser.add_argument("--image", type=str, default=None,
                        help="测试用的真实图片路径")
    args = parser.parse_args()

    test_synthetic()
    test_irregular_masks()
    test_edge_irregular_masks()

    if args.image:
        test_with_real_image(args.image)
