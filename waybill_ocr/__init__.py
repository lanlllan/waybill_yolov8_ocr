from waybill_ocr.rectifier import (
    clean_mask,
    draw_quad_on_image,
    find_quad_from_mask,
    order_points,
    perspective_transform,
    rectify_from_mask,
)

__all__ = [
    "clean_mask",
    "draw_quad_on_image",
    "find_quad_from_mask",
    "order_points",
    "perspective_transform",
    "rectify_from_mask",
    "WaybillSegmentor",
]


def __getattr__(name: str):
    if name == "WaybillSegmentor":
        from waybill_ocr.segmentor import WaybillSegmentor
        return WaybillSegmentor
    raise AttributeError(f"module 'waybill_ocr' has no attribute {name!r}")
