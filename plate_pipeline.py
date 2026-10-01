# -*- coding: utf-8 -*-
"""两阶段车牌识别流水线。

Stage 1（detect）：YOLO 输出车牌边界框；best.pt 未训练前，用 HyperLPR3
                   内置检测器输出的框兜底，保证流水线先可测。
Stage 2（recognize）：检测框 15% padding 外扩后裁剪，交给 HyperLPR3 OCR，
                      返回车牌号码、识别置信度和车牌类型。
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import hyperlpr3 as lpr3
from PIL import Image, ImageDraw, ImageFont

from config import settings


def _iou(box_a, box_b) -> float:
    """两个 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# 跨平台中文字体候选路径（按优先级）
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",                       # Windows 微软雅黑
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Debian Noto CJK
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]


def _load_font(font_size: int):
    """加载可用的中文字体；全部失败时用 PIL 默认字体（中文可能显示为方块）。"""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def put_chinese_text(image: np.ndarray, text: str, org, color=(0, 0, 255),
                     font_size: int = 22) -> np.ndarray:
    """用 PIL 在 OpenCV 图上绘制中文（cv2.putText 不支持汉字）。"""
    font = _load_font(font_size)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(org, text, font=font,
                             fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


class PlateRecognizer:
    def __init__(self, yolo_model_path: str | None = None):
        print("[Init] 加载 HyperLPR3 识别器...")
        self.lpr = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_LOW)

        self.yolo = None
        path = Path(yolo_model_path) if yolo_model_path else settings.BEST_PT
        if path and path.exists():
            from ultralytics import YOLO
            print(f"[Init] 加载 YOLO 模型: {path}")
            self.yolo = YOLO(str(path))
        else:
            print(f"[Init] 未找到 YOLO 权重（{path}），detect() 使用 HyperLPR3 内置检测兜底。")

    # ---------------- Stage 1：检测 ----------------
    def detect(self, image: np.ndarray) -> list[dict]:
        if self.yolo is not None:
            boxes = self._detect_yolo(image)
            if boxes:
                return boxes
            print("  [Detect] YOLO 无结果，改用 HyperLPR3 内置检测兜底。")
        return self._detect_lpr_fallback(image)

    def _detect_yolo(self, image: np.ndarray) -> list[dict]:
        results = self.yolo(image, verbose=False, conf=settings.YOLO_CONF)
        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.yolo.names.get(cls_id, str(cls_id))
                boxes.append({"bbox": [x1, y1, x2, y2],
                              "confidence": conf, "class": cls_name})
                print(f"  [Detect] {cls_name} @ [{x1:.0f}, {y1:.0f}, "
                      f"{x2:.0f}, {y2:.0f}] conf={conf:.2%}")
        return boxes

    def _detect_lpr_fallback(self, image: np.ndarray) -> list[dict]:
        """best.pt 就位前的兜底：直接用 HyperLPR3 的检测框（与正式路径同接口）。"""
        boxes = []
        for code, rec_conf, _plate_type, rect in self.lpr(image):
            x1, y1, x2, y2 = map(float, rect)
            boxes.append({"bbox": [x1, y1, x2, y2],
                          "confidence": float(rec_conf), "class": "car_card"})
            print(f"  [Detect-LPR] car_card @ [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] "
                  f"(置信度取OCR值) {code} conf={rec_conf:.2%}")
        return boxes

    # ---------------- Stage 2：裁剪 + OCR ----------------
    @staticmethod
    def _crop_with_padding(image: np.ndarray, bbox) -> np.ndarray | None:
        """按 PADDING_RATIO 外扩裁剪，边界保护，返回裁剪后的图片或 None。"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        pad_w = int(bw * settings.PADDING_RATIO)
        pad_h = int(bh * settings.PADDING_RATIO)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    def recognize(self, image: np.ndarray, bbox) -> dict | None:
        """裁剪→HyperLPR3 OCR。"""
        plate_crop = self._crop_with_padding(image, bbox)
        if plate_crop is None:
            return None
        ch, cw = plate_crop.shape[:2]
        print(f"  [Recognize] 裁剪图尺寸: {cw}x{ch}, "
              f"像素范围: [{plate_crop.min()}, {plate_crop.max()}]")

        results = self.lpr(plate_crop)
        if not results:
            # 二级兜底：矩形裁剪（保留透视畸变）OCR 失败时，改用整帧 LPR
            # 结果中与该框 IoU 最大者（整帧 LPR 自带四点透视矫正）。
            results = [
                r for r in self.lpr(image)
                if _iou(r[3], bbox) > 0.3
            ]
            if results:
                print("  [Recognize] 裁剪 OCR 失败，已用整帧 LPR 结果按 IoU 匹配兜底。")

        if not results:
            print("  [Recognize] 未识别到车牌")
            return None

        plate_code, confidence, plate_type, _ = max(results, key=lambda r: r[1])
        print(f"  [Recognize] -> {plate_code} conf={confidence:.2%} type={plate_type}")
        return {"plate": plate_code, "confidence": float(confidence),
                "type": int(plate_type), "crop": plate_crop}

    # ---------------- 可视化 ----------------
    @staticmethod
    def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
        vis = image.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"#{i + 1} {det['class']} {det['confidence']:.2f}"
            cv2.putText(vis, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return vis

    def save_results(self, image, detections, recognized, image_path,
                     save_dir=None):
        save_dir = Path(save_dir) if save_dir else settings.OUTPUTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        detect_path = save_dir / f"{stem}_detect.jpg"
        cv2.imwrite(str(detect_path), self.draw_detections(image, detections))

        crop_paths = []
        final = self.draw_detections(image, detections)
        for i, (det, rec) in enumerate(zip(detections, recognized), start=1):
            if rec is None:
                continue
            crop_path = save_dir / f"{stem}_plate_{i}.jpg"
            cv2.imwrite(str(crop_path), rec["crop"])
            crop_paths.append(crop_path)
            x1 = int(det["bbox"][0])
            y1 = int(det["bbox"][1])
            final = put_chinese_text(
                final, f"{rec['plate']} {rec['confidence']:.0%}",
                (x1 + 2, max(2, y1 - 30)))

        result_path = save_dir / f"{stem}_result.jpg"
        cv2.imwrite(str(result_path), final)
        print(f"[Save] {detect_path.name}；裁剪图 {len(crop_paths)} 张；{result_path.name}")
        return detect_path, crop_paths, result_path

    # ---------------- 端到端调度 ----------------
    def __call__(self, image_path, save_dir=None, save: bool = True) -> list[dict]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        h, w = image.shape[:2]
        print(f"[Pipeline] {image_path}  尺寸: {w}x{h}")

        detections = self.detect(image)
        recognized = [self.recognize(image, d["bbox"]) for d in detections]

        if save:
            self.save_results(image, detections, recognized, image_path, save_dir)

        return [
            {"plate": r["plate"], "confidence": r["confidence"],
             "type": r["type"], "bbox": d["bbox"]}
            for d, r in zip(detections, recognized) if r is not None
        ]


def main():
    parser = argparse.ArgumentParser(description="两阶段车牌识别流水线")
    parser.add_argument("images", nargs="+", help="待识别图片路径，可多张")
    parser.add_argument("--model", default=None,
                        help="YOLO 权重路径（默认 models/best.pt，缺失则 HyperLPR3 兜底）")
    parser.add_argument("--no-save", action="store_true", help="只打印结果，不保存图片")
    args = parser.parse_args()

    settings.ensure_runtime_dirs()
    recognizer = PlateRecognizer(args.model)

    for image_path in args.images:
        print("=" * 70)
        results = recognizer(image_path, save=not args.no_save)
        print("[Result]", results if results else "未识别到车牌")


if __name__ == "__main__":
    main()
