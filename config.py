# -*- coding: utf-8 -*-
"""全局统一配置：路径、MQTT、闸机、TTS 等开关。

所有模块通过 `from config import settings` 读取，禁止在业务代码里写死路径。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Settings:
    # ---------- 路径 ----------
    BASE_DIR: Path = BASE_DIR
    MODELS_DIR: Path = BASE_DIR / "models"
    BEST_PT: Path = MODELS_DIR / "best.pt"          # 阶段2训练产出
    DATASET_DIR: Path = BASE_DIR / "dataset"
    DATASET_YAML: Path = DATASET_DIR / "dataset.yaml"
    SAMPLES_DIR: Path = BASE_DIR / "samples"       # 手工测试图片
    RECORD_DIR: Path = BASE_DIR / "data"
    EXCEL_PATH: Path = RECORD_DIR / "parking_record.xlsx"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"

    # ---------- 识别参数 ----------
    YOLO_CONF: float = 0.25        # YOLO 置信度阈值
    PADDING_RATIO: float = 0.15    # OCR 裁剪外扩比例
    CLASS_NAMES = {0: "license_plate"}  # YOLO 类别（与数据集一致）

    # ---------- 车位 ----------
    TOTAL_SPACES: int = 24
    SPACE_ROWS = ("A", "B", "C", "D")
    SPACE_COLS = 6

    @property
    def ALL_SPACES(self) -> list[str]:
        return [f"{r}{c}" for r in self.SPACE_ROWS for c in range(1, self.SPACE_COLS + 1)]

    # ---------- MQTT / 闸机 ----------
    MQTT_BROKER: str = os.environ.get("MQTT_BROKER", "localhost")
    MQTT_PORT: int = int(os.environ.get("MQTT_PORT", "1883"))
    MQTT_USERNAME: str | None = None     # 本机 Mosquitto 默认匿名
    MQTT_PASSWORD: str | None = None
    MQTT_PUB_TOPIC: str = "/sys/thing/s2c/msg"
    GATE_DELAY: int = 3                  # 抬杆后等待秒数
    SERVO_OPEN = {"Servo": 4}
    SERVO_CLOSE = {"Servo": 0}

    # ---------- TTS ----------
    TTS_ENABLED: bool = True
    TTS_VOICE: str = "zh-CN-XiaoxiaoNeural"
    TTS_DIR: Path = BASE_DIR / "data" / "tts_cache"

    # ---------- 服务 ----------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CAMERA_INDEX: int = 0

    def ensure_runtime_dirs(self) -> None:
        """创建运行时目录（模型/数据集等源码目录不动）。"""
        for d in (
            self.MODELS_DIR,
            self.SAMPLES_DIR,
            self.RECORD_DIR,
            self.UPLOADS_DIR,
            self.OUTPUTS_DIR,
            self.TTS_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
