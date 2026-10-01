# -*- coding: utf-8 -*-
"""扩展识别准确率测试：用 test 集大图，双方法交叉验证。

方法：
- 对每张 test 图片同时跑 YOLO11n+OCR（方案A）和 HyperLPR3 pipeline（方案B）
- 两者都检测到车牌且字符串完全一致 → 视为高置信正确（两个独立 OCR 引擎一致，错误概率极低）
- 统计：检测率、一致率、以一致结果为基准的各自准确率
"""
import json
import time
from pathlib import Path

import cv2

import hyperlpr3 as lpr3
from ultralytics import YOLO

from config import settings
from plate_pipeline import PlateRecognizer

TEST_IMG_DIR = settings.DATASET_DIR / "test" / "images"
N_SAMPLES = 40  # 取 40 张做统计


def main():
    print("[Init] 加载模型...")
    pr = PlateRecognizer()
    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_LOW)

    img_files = sorted(TEST_IMG_DIR.glob("*.jpg"))[:N_SAMPLES]
    print(f"测试图片数: {len(img_files)}")

    stats = {
        "total": len(img_files),
        "yolo_detected": 0,
        "lpr_detected": 0,
        "both_detected": 0,
        "agreed": 0,
        "disagreed": 0,
    }
    yolo_only_correct = 0  # 以一致结果为基准
    lpr_only_correct = 0
    yolo_times, lpr_times = [], []
    disagreements = []

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 方案A：YOLO11n + 裁剪 OCR
        t0 = time.perf_counter()
        res_a = pr(str(img_path), save=False)
        yolo_times.append(time.perf_counter() - t0)
        plate_a = res_a[0]["plate"] if res_a else None

        # 方案B：HyperLPR3 pipeline
        t0 = time.perf_counter()
        res_b = catcher.pipeline(img)
        lpr_times.append(time.perf_counter() - t0)
        plate_b = res_b[0][0] if res_b else None

        if plate_a:
            stats["yolo_detected"] += 1
        if plate_b:
            stats["lpr_detected"] += 1

        if plate_a and plate_b:
            stats["both_detected"] += 1
            if plate_a == plate_b:
                stats["agreed"] += 1
                yolo_only_correct += 1
                lpr_only_correct += 1
            else:
                stats["disagreed"] += 1
                disagreements.append({
                    "file": img_path.name,
                    "yolo": plate_a,
                    "lpr": plate_b,
                })
                # 以 HyperLPR3 为基准（它在对比实验中精度更高）
                # 但这里不强行选边，只记录分歧
        elif plate_a and not plate_b:
            yolo_only_correct += 1  # YOLO 检出而 LPR 漏检，以 YOLO 为准
        elif plate_b and not plate_a:
            lpr_only_correct += 1  # LPR 检出而 YOLO 漏检，以 LPR 为准

    n = stats["total"]
    print(f"\n===== 识别准确率扩展测试（{n} 张 test 集图片）=====")
    print(f"YOLO11n+OCR 检出率: {stats['yolo_detected']}/{n} = {stats['yolo_detected']/n:.1%}")
    print(f"HyperLPR3 检出率:  {stats['lpr_detected']}/{n} = {stats['lpr_detected']/n:.1%}")
    print(f"两者都检出: {stats['both_detected']}/{n}")
    print(f"结果一致: {stats['agreed']}/{stats['both_detected']} = "
          f"{stats['agreed']/stats['both_detected']:.1%}" if stats["both_detected"] else "")
    print(f"结果分歧: {stats['disagreed']}")

    # 以"一致结果"为高置信真值，计算各自准确率
    # 对分歧样本，无法确定谁对，保守估计：各自准确率 = 一致数 / 总检出数
    yolo_acc = stats["agreed"] / stats["yolo_detected"] if stats["yolo_detected"] else 0
    lpr_acc = stats["agreed"] / stats["lpr_detected"] if stats["lpr_detected"] else 0
    print(f"\n以一致结果为基准的准确率（保守估计）:")
    print(f"  YOLO11n+OCR: {yolo_acc:.1%} ({stats['agreed']}/{stats['yolo_detected']})")
    print(f"  HyperLPR3:   {lpr_acc:.1%} ({stats['agreed']}/{stats['lpr_detected']})")

    import numpy as np
    print(f"\n端到端延迟:")
    print(f"  YOLO11n+OCR: {np.mean(yolo_times)*1000:.1f}ms")
    print(f"  HyperLPR3:   {np.mean(lpr_times)*1000:.1f}ms")

    if disagreements:
        print(f"\n分歧样本（共 {len(disagreements)} 个）:")
        for d in disagreements[:10]:
            print(f"  {d['file']}: YOLO={d['yolo']} | LPR={d['lpr']}")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": n,
        "yolo_detect_rate": stats["yolo_detected"] / n,
        "lpr_detect_rate": stats["lpr_detected"] / n,
        "both_detected": stats["both_detected"],
        "agreement_rate": stats["agreed"] / stats["both_detected"] if stats["both_detected"] else 0,
        "yolo_accuracy_vs_agreement": yolo_acc,
        "lpr_accuracy_vs_agreement": lpr_acc,
        "yolo_latency_ms": float(np.mean(yolo_times) * 1000),
        "lpr_latency_ms": float(np.mean(lpr_times) * 1000),
        "disagreements": disagreements,
    }
    out = settings.OUTPUTS_DIR / "recognition_accuracy_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
