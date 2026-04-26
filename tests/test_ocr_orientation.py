"""Tests for OCR orientation selection logic."""

from __future__ import annotations

import numpy as np

from waybill_ocr.ocr_engine import WaybillOCR


def test_find_best_orientation_uses_zero_degree_as_shared_threshold(monkeypatch):
    """A later candidate should win if it beats 0 deg threshold and previous non-zero candidates."""
    scores = iter([10.0, 16.0, 20.0, 12.0])

    def fake_score(self, image, conf_threshold=0.7):
        return next(scores)

    monkeypatch.setattr(WaybillOCR, "_score_orientation", fake_score)
    ocr = WaybillOCR.__new__(WaybillOCR)
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    assert ocr.find_best_orientation(image) == 90


def test_find_best_orientation_keeps_zero_when_candidates_do_not_clear_threshold(monkeypatch):
    scores = iter([10.0, 14.0, 14.9, 13.0])

    def fake_score(self, image, conf_threshold=0.7):
        return next(scores)

    monkeypatch.setattr(WaybillOCR, "_score_orientation", fake_score)
    ocr = WaybillOCR.__new__(WaybillOCR)
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    assert ocr.find_best_orientation(image) == 0
