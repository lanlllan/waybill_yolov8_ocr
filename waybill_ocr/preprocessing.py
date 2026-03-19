"""
图像预处理模块：优化 OCR 识别质量。

职责：
    1. 自适应二值化（提升文字对比度）
    2. 去噪处理（减少图像噪声）
    3. 对比度增强（CLAHE）
    4. 锐化处理（增强边缘）
    5. 亮度均衡化

策略：
    - 提供多种预处理方法，可独立或组合使用
    - 每种方法都有强度参数，可根据图像质量调整
    - 保留原始色彩空间，避免信息损失
"""

from __future__ import annotations

import cv2
import numpy as np


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    使用 CLAHE（自适应直方图均衡化）增强对比度。

    特别适用于光照不均匀的图像，可以提升文字区域的可读性。

    Args:
        image: 输入图像（BGR 或灰度）
        clip_limit: 对比度限制阈值，越大对比度越强（推荐 2.0-3.0）
        tile_size: 网格大小，用于局部对比度增强（推荐 8）

    Returns:
        对比度增强后的图像
    """
    if len(image.shape) == 2:
        # 灰度图直接处理
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)

    # 彩色图：转换到 LAB 色彩空间，只处理亮度通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    对图像进行去噪处理。

    使用非局部均值去噪算法，可以在去除噪声的同时保留边缘细节。

    Args:
        image: 输入图像（BGR 或灰度）
        strength: 去噪强度（推荐 5-15，值越大去噪越强但可能模糊细节）

    Returns:
        去噪后的图像
    """
    if len(image.shape) == 2:
        # 灰度图
        return cv2.fastNlMeansDenoising(image, None, h=strength, templateWindowSize=7, searchWindowSize=21)
    else:
        # 彩色图
        return cv2.fastNlMeansDenoisingColored(image, None, h=strength, hColor=strength,
                                                templateWindowSize=7, searchWindowSize=21)


def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    对图像进行锐化处理，增强边缘和细节。

    使用 Unsharp Masking 方法，可以提升文字边缘的清晰度。

    Args:
        image: 输入图像（BGR 或灰度）
        strength: 锐化强度（推荐 0.5-2.0，值越大锐化越明显）

    Returns:
        锐化后的图像
    """
    # 高斯模糊
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    # Unsharp Masking: 原图 + strength * (原图 - 模糊图)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def binarize_adaptive(image: np.ndarray, block_size: int = 15, c: int = 10) -> np.ndarray:
    """
    自适应二值化处理。

    对于文字识别，二值化可以显著提升 OCR 准确率，特别是光照不均匀的情况。

    Args:
        image: 输入图像（BGR 或灰度）
        block_size: 自适应阈值的邻域大小（必须为奇数，推荐 11-25）
        c: 从平均值中减去的常数（推荐 5-15）

    Returns:
        二值化后的灰度图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 自适应高斯阈值
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )
    return binary


def adjust_brightness(image: np.ndarray, target_brightness: int = 128) -> np.ndarray:
    """
    调整图像亮度到目标值。

    对于过暗或过亮的图像，调整到合适的亮度范围可以提升 OCR 效果。

    Args:
        image: 输入图像（BGR 或灰度）
        target_brightness: 目标亮度值（0-255，推荐 120-140）

    Returns:
        亮度调整后的图像
    """
    if len(image.shape) == 3:
        # 彩色图：在 HSV 空间调整 V 通道
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 计算当前平均亮度
        current_brightness = np.mean(v)

        # 计算调整系数
        if current_brightness > 0:
            alpha = target_brightness / current_brightness
            v = np.clip(v * alpha, 0, 255).astype(np.uint8)

        adjusted = cv2.merge([h, s, v])
        return cv2.cvtColor(adjusted, cv2.COLOR_HSV2BGR)
    else:
        # 灰度图：直接调整
        current_brightness = np.mean(image)
        if current_brightness > 0:
            alpha = target_brightness / current_brightness
            return np.clip(image * alpha, 0, 255).astype(np.uint8)
        return image


def preprocess_for_ocr(
    image: np.ndarray,
    enhance_contrast: bool = True,
    denoise: bool = True,
    sharpen: bool = False,
    adjust_brightness_flag: bool = False,
    binarize: bool = False,
    contrast_clip_limit: float = 2.0,
    denoise_strength: int = 7,
    sharpen_strength: float = 0.5,
    target_brightness: int = 130,
) -> np.ndarray:
    """
    综合预处理流程，优化图像用于 OCR 识别。

    处理顺序：
    1. 亮度调整（可选）
    2. 去噪（可选）
    3. 对比度增强（推荐）
    4. 锐化（可选）
    5. 二值化（可选）

    Args:
        image: 输入图像（BGR）
        enhance_contrast: 是否增强对比度（推荐开启）
        denoise: 是否去噪（推荐开启）
        sharpen: 是否锐化（低质量图像可开启）
        adjust_brightness_flag: 是否调整亮度（过暗/过亮图像可开启）
        binarize: 是否二值化（根据场景决定，可能损失颜色信息）
        contrast_clip_limit: CLAHE 对比度限制
        denoise_strength: 去噪强度
        sharpen_strength: 锐化强度
        target_brightness: 目标亮度

    Returns:
        预处理后的图像
    """
    result = image.copy()

    # 1. 亮度调整（如果需要）
    if adjust_brightness_flag:
        result = adjust_brightness(result, target_brightness)

    # 2. 去噪（减少噪声干扰）
    if denoise:
        result = denoise_image(result, strength=denoise_strength)

    # 3. 对比度增强（提升文字可读性）
    if enhance_contrast:
        result = enhance_contrast(result, clip_limit=contrast_clip_limit)

    # 4. 锐化（增强边缘）
    if sharpen:
        result = sharpen_image(result, strength=sharpen_strength)

    # 5. 二值化（可选，转为黑白图）
    if binarize:
        result = binarize_adaptive(result)
        # 如果二值化后是灰度图，转回 BGR 以保持一致性
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result


def auto_preprocess(image: np.ndarray) -> np.ndarray:
    """
    自动预处理模式：根据图像质量自动选择合适的预处理方法。

    评估标准：
    - 亮度：过暗或过亮则调整
    - 对比度：低对比度则增强
    - 噪声：总是应用轻度去噪

    Args:
        image: 输入图像（BGR）

    Returns:
        自动预处理后的图像
    """
    # 转换到灰度计算统计信息
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 评估亮度
    mean_brightness = np.mean(gray)
    adjust_brightness_flag = mean_brightness < 80 or mean_brightness > 180

    # 评估对比度（使用标准差作为对比度指标）
    contrast = np.std(gray)
    enhance_contrast_flag = contrast < 40

    # 应用预处理
    return preprocess_for_ocr(
        image,
        enhance_contrast=enhance_contrast_flag or True,  # 总是增强对比度
        denoise=True,  # 总是轻度去噪
        sharpen=False,  # 避免过度锐化
        adjust_brightness_flag=adjust_brightness_flag,
        binarize=False,  # 保留颜色信息
        denoise_strength=5,  # 轻度去噪
        contrast_clip_limit=2.5 if enhance_contrast_flag else 2.0,
    )
