# -*- coding: utf-8 -*-
"""pytest 共享 fixtures。"""
import shutil
from pathlib import Path

import pytest

from config import settings


@pytest.fixture(scope="session")
def recognizer():
    """全局共享一个识别器（加载模型开销大）。"""
    from plate_pipeline import PlateRecognizer

    return PlateRecognizer()


@pytest.fixture()
def isolated_excel(tmp_path, monkeypatch):
    """把 Excel 路径指向临时文件，避免污染真实数据。"""
    tmp_excel = tmp_path / "parking_record.xlsx"
    monkeypatch.setattr(settings, "EXCEL_PATH", tmp_excel)
    monkeypatch.setattr(settings, "RECORD_DIR", tmp_path)
    return tmp_excel


@pytest.fixture()
def sample_images():
    """可用的测试图片清单。"""
    return {
        "front": "samples/sample2.jpg",            # 沪AD07979
        "dataset_train": "samples/ds_train_9799850d72.jpg",  # 皖AD05119
        "dataset_val": "samples/ds_val_48f2297dbd.jpg",      # 皖AF06700
        "tilted": "samples/sample_blue1.jpg",      # 苏ED5172（倾斜）
        "blank": "outputs/blank.jpg",              # 无车牌
    }
