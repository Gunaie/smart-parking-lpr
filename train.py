# -*- coding: utf-8 -*-
"""YOLO11n 车牌检测训练脚本（CPU 可跑）。

用法：
    python train.py                         # 用默认参数训练
    python train.py --epochs 50 --batch 8   # 自定义
训练产物在 runs/detect/train/weights/best.pt，
脚本结束后会自动复制到 models/best.pt 供识别流水线加载。
"""
import argparse
import shutil

from ultralytics import YOLO

from config import settings


def main():
    parser = argparse.ArgumentParser(description="YOLO11n 车牌检测训练")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练权重")
    parser.add_argument("--data", default=str(settings.DATASET_YAML),
                        help="数据集配置 dataset.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8,
                        help="CPU 建议 4~16，按内存调整")
    parser.add_argument("--device", default="cpu", help="cpu / 0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="train", help="runs/detect 下的运行名")
    args = parser.parse_args()

    settings.ensure_runtime_dirs()
    print(f"[Train] 模型={args.model} 数据={args.data} epochs={args.epochs} "
          f"imgsz={args.imgsz} batch={args.batch} device={args.device}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        exist_ok=True,
        verbose=True,
        plots=False,
    )

    save_dir = getattr(results, "save_dir", None)
    best_src = (save_dir / "weights" / "best.pt") if save_dir else None
    if best_src and best_src.exists():
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, settings.BEST_PT)
        print(f"[Train] 已复制 {best_src} -> {settings.BEST_PT}")
    else:
        print("[Train] 警告：未找到 best.pt，请检查训练输出目录。")


if __name__ == "__main__":
    main()
