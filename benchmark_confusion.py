# -*- coding: utf-8 -*-
"""字符级混淆矩阵分析。

方法：
- 遍历 test 集图片，同时跑 YOLO+OCR（预测）和 HyperLPR3（参考真值）
- 两者都检出且长度一致时，逐字符统计混淆
- 重点绘制第 1 位（省份汉字）混淆矩阵热力图
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt

import hyperlpr3 as lpr3

from config import settings
from plate_pipeline import PlateRecognizer

TEST_IMG_DIR = settings.DATASET_DIR / "test" / "images"
PROVINCES = list("京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")  # 31 省


def main():
    print("[Init] 加载模型...")
    pr = PlateRecognizer()
    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_LOW)

    img_files = sorted(TEST_IMG_DIR.glob("*.jpg"))
    print(f"测试图片数: {len(img_files)}")

    # 省份汉字混淆矩阵：行=参考真值，列=预测
    prov_conf = defaultdict(lambda: defaultdict(int))
    # 全部字符混淆（第 2~7 位）
    char_conf = defaultdict(lambda: defaultdict(int))

    total = 0
    matched = 0  # 完全一致
    both_detected = 0

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        res_a = pr(str(img_path), save=False)
        plate_a = res_a[0]["plate"] if res_a else None

        res_b = catcher.pipeline(img)
        plate_b = res_b[0][0] if res_b else None

        if not plate_a or not plate_b:
            continue
        both_detected += 1

        # 只分析长度一致的样本（大部分为 7 位蓝牌）
        if len(plate_a) != len(plate_b):
            continue
        total += 1

        if plate_a == plate_b:
            matched += 1

        # 第 1 位：省份汉字
        true_p = plate_b[0]
        pred_p = plate_a[0]
        if true_p in PROVINCES and pred_p in PROVINCES:
            prov_conf[true_p][pred_p] += 1

        # 第 2~7 位：字母数字
        for i in range(1, min(len(plate_a), 7)):
            char_conf[plate_b[i]][plate_a[i]] += 1

    print(f"\n===== 混淆矩阵分析 =====")
    print(f"两者都检出: {both_detected}")
    print(f"长度一致样本: {total}")
    print(f"完全一致: {matched}/{total} = {matched/total:.1%}" if total else "")

    # 省份混淆矩阵
    prov_labels = sorted({k for d in prov_conf.values() for k in d} | set(prov_conf.keys()))
    if prov_labels:
        n = len(prov_labels)
        mat = np.zeros((n, n), dtype=int)
        idx = {c: i for i, c in enumerate(prov_labels)}
        for true_c, preds in prov_conf.items():
            for pred_c, cnt in preds.items():
                mat[idx[true_c]][idx[pred_c]] = cnt

        # 保存矩阵数据
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "both_detected": both_detected,
            "length_matched_samples": total,
            "exact_match_rate": matched / total if total else 0,
            "province_labels": prov_labels,
            "province_confusion_matrix": mat.tolist(),
        }
        out_json = settings.OUTPUTS_DIR / "confusion_matrix.json"
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"矩阵数据已保存: {out_json}")

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(prov_labels, fontsize=9)
        ax.set_yticklabels(prov_labels, fontsize=9)
        ax.set_xlabel("Predicted (YOLO+OCR)", fontsize=11)
        ax.set_ylabel("True (HyperLPR3)", fontsize=11)
        ax.set_title(f"Province Character Confusion Matrix (n={total})", fontsize=12)

        # 标注数字
        for i in range(n):
            for j in range(n):
                if mat[i][j] > 0:
                    color = "white" if mat[i][j] > mat.max() / 2 else "black"
                    ax.text(j, i, str(mat[i][j]), ha="center", va="center",
                            color=color, fontsize=8)

        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        out_png = settings.OUTPUTS_DIR / "confusion_matrix_province.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"热力图已保存: {out_png}")

        # 打印易混淆对
        print("\n易混淆省份（真值→预测）:")
        pairs = []
        for true_c, preds in prov_conf.items():
            for pred_c, cnt in preds.items():
                if true_c != pred_c:
                    pairs.append((true_c, pred_c, cnt))
        pairs.sort(key=lambda x: -x[2])
        for t, p, c in pairs[:10]:
            print(f"  {t} -> {p}: {c} 次")

    # 字母数字 Top 混淆
    if char_conf:
        print("\n字母数字 Top 混淆（真值→预测）:")
        pairs = []
        for true_c, preds in char_conf.items():
            for pred_c, cnt in preds.items():
                if true_c != pred_c:
                    pairs.append((true_c, pred_c, cnt))
        pairs.sort(key=lambda x: -x[2])
        for t, p, c in pairs[:10]:
            print(f"  {t} -> {p}: {c} 次")


if __name__ == "__main__":
    main()
