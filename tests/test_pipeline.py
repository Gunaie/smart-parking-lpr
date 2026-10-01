# -*- coding: utf-8 -*-
"""识别流水线精度回归测试。"""
import cv2
import numpy as np
import pytest

from config import settings


class TestPipeline:
    """plate_pipeline 识别精度回归。"""

    def test_frontal_plate(self, recognizer, sample_images):
        """正面近景车牌：高置信度识别。"""
        results = recognizer(sample_images["front"], save=False)
        assert len(results) == 1
        assert results[0]["plate"] == "沪AD07979"
        assert results[0]["confidence"] > 0.95

    def test_dataset_image_train(self, recognizer, sample_images):
        """数据集样本 1。"""
        results = recognizer(sample_images["dataset_train"], save=False)
        assert len(results) == 1
        assert results[0]["plate"] == "皖AD05119"

    def test_dataset_image_val(self, recognizer, sample_images):
        """数据集样本 2。"""
        results = recognizer(sample_images["dataset_val"], save=False)
        assert len(results) == 1
        assert results[0]["plate"] == "皖AF06700"

    def test_tilted_plate_fallback(self, recognizer, sample_images):
        """极端倾斜图：YOLO 可能漏检，走 LPR 检测兜底 + IoU 兜底。"""
        results = recognizer(sample_images["tilted"], save=False)
        assert len(results) >= 1
        plates = {r["plate"] for r in results}
        assert "苏ED5172" in plates

    def test_blank_image(self, recognizer, sample_images):
        """无车牌图应返回空列表。"""
        results = recognizer(sample_images["blank"], save=False)
        assert results == []

    def test_detect_returns_bbox(self, recognizer, sample_images):
        """检测框应在图像范围内。"""
        img = cv2.imread(sample_images["front"])
        dets = recognizer.detect(img)
        h, w = img.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            assert 0 <= x1 < x2 <= w
            assert 0 <= y1 < y2 <= h


class TestPadding:
    """边界保护逻辑。"""

    def test_crop_does_not_exceed(self, recognizer):
        """角落车牌裁剪不越界。"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = recognizer._crop_with_padding(img, [0, 0, 10, 10])
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        assert crop.shape[0] <= 100 and crop.shape[1] <= 100
