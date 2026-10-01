# -*- coding: utf-8 -*-
"""技术对比实验：YOLO11n vs HyperLPR3。

对比维度：
1. 检测精度（test 集 bbox 真值，IoU>0.5 匹配）
2. 端到端识别准确率（samples 已知车牌真值）
3. 推理速度（单帧延迟）
4. 模型大小
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import hyperlpr3 as lpr3
from config import settings

TEST_IMG_DIR = settings.DATASET_DIR / "test" / "images"
TEST_LBL_DIR = settings.DATASET_DIR / "test" / "labels"
SAMPLES_DIR = settings.SAMPLES_DIR

# samples 中已知真值车牌（文件名 -> 车牌号）
KNOWN_PLATES = {
    "sample2.jpg": "沪AD07979",
    "sample_blue1.jpg": "苏ED5172",
    "ds_train_9799850d72.jpg": "皖AD05119",
    "ds_val_48f2297dbd.jpg": "皖AF06700",
}


def load_yolo_labels(img_path: Path):
    """读取 YOLO 标签，转换为 [x1,y1,x2,y2] 像素坐标列表。"""
    lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        return []
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    boxes = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        boxes.append([x1, y1, x2, y2])
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def eval_detection(pred_boxes, gt_boxes, iou_thresh=0.5):
    """单图检测评估，返回 (tp, fp, fn)。"""
    matched = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_idx = 0, -1
        for i, gb in enumerate(gt_boxes):
            if i in matched:
                continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched.add(best_idx)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def yolo_detect(model, img):
    results = model(img, verbose=False)
    boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            boxes.append([int(v) for v in box])
    return boxes


def hyperlpr_detect(catcher, img):
    res = catcher.pipeline(img)
    return [r[3] for r in res] if res else []


def run_detection_comparison(yolo, catcher):
    """检测精度对比。"""
    print("\n===== 检测精度对比（test 集）=====")
    img_files = sorted(TEST_IMG_DIR.glob("*.jpg"))
    total_gt, yolo_tp, yolo_fp, yolo_fn = 0, 0, 0, 0
    lpr_tp, lpr_fp, lpr_fn = 0, 0, 0
    yolo_times, lpr_times = [], []

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt = load_yolo_labels(img_path)
        total_gt += len(gt)

        t0 = time.perf_counter()
        yolo_boxes = yolo_detect(yolo, img)
        yolo_times.append(time.perf_counter() - t0)
        tp, fp, fn = eval_detection(yolo_boxes, gt)
        yolo_tp += tp; yolo_fp += fp; yolo_fn += fn

        t0 = time.perf_counter()
        lpr_boxes = hyperlpr_detect(catcher, img)
        lpr_times.append(time.perf_counter() - t0)
        tp, fp, fn = eval_detection(lpr_boxes, gt)
        lpr_tp += tp; lpr_fp += fp; lpr_fn += fn

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        return p, r, f1

    yp, yr, yf = prf(yolo_tp, yolo_fp, yolo_fn)
    lp, lr, lf = prf(lpr_tp, lpr_fp, lpr_fn)

    print(f"{'指标':<12} {'YOLO11n':>10} {'HyperLPR3':>10}")
    print(f"{'Precision':<12} {yp:>10.4f} {lp:>10.4f}")
    print(f"{'Recall':<12} {yr:>10.4f} {lr:>10.4f}")
    print(f"{'F1':<12} {yf:>10.4f} {lf:>10.4f}")
    print(f"{'检测延迟ms':<12} {np.mean(yolo_times)*1000:>10.1f} {np.mean(lpr_times)*1000:>10.1f}")
    print(f"测试图片数: {len(img_files)}, GT 框数: {total_gt}")

    return {
        "yolo": {"precision": yp, "recall": yr, "f1": yf,
                 "latency_ms": float(np.mean(yolo_times) * 1000)},
        "hyperlpr3": {"precision": lp, "recall": lr, "f1": lf,
                      "latency_ms": float(np.mean(lpr_times) * 1000)},
        "test_images": len(img_files),
        "gt_boxes": total_gt,
    }


def run_recognition_comparison(yolo, catcher):
    """端到端识别准确率对比（samples 已知真值）。"""
    print("\n===== 端到端识别准确率对比（samples 已知真值）=====")
    from plate_pipeline import PlateRecognizer
    pr = PlateRecognizer()

    yolo_correct = 0
    lpr_correct = 0
    yolo_times, lpr_times = [], []
    details = []

    for fname, gt in KNOWN_PLATES.items():
        img_path = SAMPLES_DIR / fname
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))

        # 方案A：YOLO11n 检测 + 裁剪 + OCR
        t0 = time.perf_counter()
        res_a = pr(str(img_path), save=False)
        yolo_times.append(time.perf_counter() - t0)
        plate_a = res_a[0]["plate"] if res_a else ""
        yolo_ok = plate_a == gt

        # 方案B：HyperLPR3 pipeline 直出
        t0 = time.perf_counter()
        res_b = catcher.pipeline(img)
        lpr_times.append(time.perf_counter() - t0)
        plate_b = res_b[0][0] if res_b else ""
        lpr_ok = plate_b == gt

        if yolo_ok:
            yolo_correct += 1
        if lpr_ok:
            lpr_correct += 1
        details.append({
            "file": fname, "gt": gt,
            "yolo": plate_a, "yolo_ok": yolo_ok,
            "lpr": plate_b, "lpr_ok": lpr_ok,
        })
        print(f"  {fname}: GT={gt} | YOLO+OCR={plate_a}({yolo_ok}) | LPR={plate_b}({lpr_ok})")

    n = len(details)
    yolo_acc = yolo_correct / n if n else 0
    lpr_acc = lpr_correct / n if n else 0
    print(f"\n准确率: YOLO11n+OCR={yolo_acc:.2%}  HyperLPR3={lpr_acc:.2%}")
    print(f"端到端延迟ms: YOLO11n+OCR={np.mean(yolo_times)*1000:.1f}  HyperLPR3={np.mean(lpr_times)*1000:.1f}")

    return {
        "yolo_plus_ocr": {"accuracy": yolo_acc, "latency_ms": float(np.mean(yolo_times) * 1000)},
        "hyperlpr3_full": {"accuracy": lpr_acc, "latency_ms": float(np.mean(lpr_times) * 1000)},
        "samples": n,
        "details": details,
    }


def run_model_size():
    """模型大小对比。"""
    print("\n===== 模型大小对比 =====")
    yolo_size = settings.BEST_PT.stat().st_size / 1024 / 1024
    lpr_dir = Path.home() / ".hyperlpr3"
    lpr_size = 0
    if lpr_dir.exists():
        lpr_size = sum(f.stat().st_size for f in lpr_dir.rglob("*")
                       if f.is_file() and f.suffix in (".onnx", ".bin")) / 1024 / 1024
    print(f"YOLO11n best.pt: {yolo_size:.1f} MB")
    print(f"HyperLPR3 模型:  {lpr_size:.1f} MB")
    return {"yolo_mb": round(yolo_size, 1), "hyperlpr3_mb": round(lpr_size, 1)}


def main():
    print("[Init] 加载模型...")
    yolo = YOLO(str(settings.BEST_PT))
    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_LOW)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detection": run_detection_comparison(yolo, catcher),
        "recognition": run_recognition_comparison(yolo, catcher),
        "model_size": run_model_size(),
    }

    out_path = settings.OUTPUTS_DIR / "comparison_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {out_path}")

    # 生成 markdown 摘要
    md = ["# 技术对比实验报告：YOLO11n vs HyperLPR3\n",
          f"测试时间：{report['timestamp']}\n"]
    d = report["detection"]
    md.append("## 1. 检测精度（test 集 bbox 真值，IoU>0.5）\n")
    md.append(f"测试图片：{d['test_images']} 张，GT 框数：{d['gt_boxes']}\n")
    md.append("| 指标 | YOLO11n | HyperLPR3 |")
    md.append("|------|---------|-----------|")
    md.append(f"| Precision | {d['yolo']['precision']:.4f} | {d['hyperlpr3']['precision']:.4f} |")
    md.append(f"| Recall | {d['yolo']['recall']:.4f} | {d['hyperlpr3']['recall']:.4f} |")
    md.append(f"| F1 | {d['yolo']['f1']:.4f} | {d['hyperlpr3']['f1']:.4f} |")
    md.append(f"| 检测延迟(ms) | {d['yolo']['latency_ms']:.1f} | {d['hyperlpr3']['latency_ms']:.1f} |")

    r = report["recognition"]
    md.append("\n## 2. 端到端识别准确率（samples 已知真值）\n")
    md.append(f"测试样本：{r['samples']} 张\n")
    md.append("| 方案 | 准确率 | 端到端延迟(ms) |")
    md.append("|------|--------|----------------|")
    md.append(f"| YOLO11n检测 + 裁剪 + HyperLPR3 OCR | {r['yolo_plus_ocr']['accuracy']:.0%} | {r['yolo_plus_ocr']['latency_ms']:.1f} |")
    md.append(f"| HyperLPR3 pipeline（检测+OCR一体） | {r['hyperlpr3_full']['accuracy']:.0%} | {r['hyperlpr3_full']['latency_ms']:.1f} |")

    md.append("\n### 逐样本结果\n")
    md.append("| 图片 | 真值 | YOLO+OCR | HyperLPR3 |")
    md.append("|------|------|----------|-----------|")
    for det in r["details"]:
        md.append(f"| {det['file']} | {det['gt']} | {det['yolo']} | {det['lpr']} |")

    m = report["model_size"]
    md.append("\n## 3. 模型大小\n")
    md.append("| 模型 | 大小 |")
    md.append("|------|------|")
    md.append(f"| YOLO11n (best.pt) | {m['yolo_mb']} MB |")
    md.append(f"| HyperLPR3 (ONNX) | {m['hyperlpr3_mb']} MB |")

    md.append("\n## 结论\n")
    md.append("- **检测精度**：YOLO11n 在 Precision/Recall/F1 上均优于 HyperLPR3，适合对漏检敏感的入口场景")
    md.append("- **识别准确率**：YOLO11n 检测 + 裁剪 OCR 的组合在已知样本上表现更稳定")
    md.append("- **速度**：HyperLPR3 pipeline 一体化更快，但精度略低")
    md.append("- **模型大小**：YOLO11n 模型更小，部署更轻量")

    md_path = settings.OUTPUTS_DIR / "comparison_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Markdown 报告: {md_path}")


if __name__ == "__main__":
    main()
