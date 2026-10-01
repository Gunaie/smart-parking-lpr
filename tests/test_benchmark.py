# -*- coding: utf-8 -*-
"""性能基准测试：识别延迟、API 响应时间、内存占用。"""
import json
import time
import tracemalloc
from pathlib import Path

import pytest
import requests

from config import settings

BASE = "http://localhost:8765"


class TestPerformance:
    """性能指标基线，用于回归对比。"""

    def test_recognition_latency(self, recognizer, sample_images):
        """单张图片端到端识别延迟。"""
        times = []
        for path in [sample_images["front"], sample_images["dataset_train"],
                     sample_images["dataset_val"]]:
            t0 = time.perf_counter()
            recognizer(path, save=False)
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        print(f"\n  [Latency] 单张识别: {avg:.3f}s (min={min(times):.3f}, max={max(times):.3f})")
        assert avg < 5.0, f"识别太慢: {avg:.3f}s"

    def test_api_entry_latency(self, sample_images):
        """API 进场接口端到端延迟。"""
        requests.post(f"{BASE}/reset")
        times = []
        for _ in range(3):
            with open(sample_images["front"], "rb") as f:
                t0 = time.perf_counter()
                r = requests.post(f"{BASE}/entry", files={"file": f})
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                requests.post(f"{BASE}/reset")
        avg = sum(times) / len(times)
        print(f"\n  [API Latency] 进场: {avg:.3f}s")
        assert avg < 5.0, f"API 太慢: {avg:.3f}s"

    def test_memory_peak(self, recognizer, sample_images):
        """识别峰值内存。"""
        tracemalloc.start()
        recognizer(sample_images["front"], save=False)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024
        print(f"\n  [Memory] 峰值: {peak_mb:.1f}MB")
        assert peak_mb < 2000, f"内存过高: {peak_mb:.1f}MB"

    def test_model_file_size(self):
        """模型文件大小。"""
        size_mb = settings.BEST_PT.stat().st_size / 1024 / 1024
        print(f"\n  [Model] 大小: {size_mb:.1f}MB")
        assert size_mb < 50, f"模型过大: {size_mb:.1f}MB"

    def test_benchmark_report(self, recognizer, sample_images):
        """生成完整基准报告。"""
        report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": {}}

        # 识别延迟
        times = []
        for key in ["front", "dataset_train", "dataset_val"]:
            t0 = time.perf_counter()
            recognizer(sample_images[key], save=False)
            times.append(time.perf_counter() - t0)
        report["metrics"]["recognition_latency_s"] = {
            "avg": round(sum(times) / len(times), 3),
            "min": round(min(times), 3),
            "max": round(max(times), 3),
        }

        # 模型大小
        report["metrics"]["model_size_mb"] = round(
            settings.BEST_PT.stat().st_size / 1024 / 1024, 1)

        # API 延迟
        requests.post(f"{BASE}/reset")
        api_times = []
        for _ in range(3):
            with open(sample_images["front"], "rb") as f:
                t0 = time.perf_counter()
                requests.post(f"{BASE}/entry", files={"file": f})
                api_times.append(time.perf_counter() - t0)
                requests.post(f"{BASE}/reset")
        report["metrics"]["api_entry_latency_s"] = round(
            sum(api_times) / len(api_times), 3)

        report_path = settings.OUTPUTS_DIR / "benchmark_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\n  [Report] 已保存: {report_path}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
