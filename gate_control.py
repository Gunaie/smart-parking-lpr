# -*- coding: utf-8 -*-
"""闸机控制：通过 MQTT 向实训箱/模拟器发送舵机抬杆/落杆指令。

Topic: /sys/thing/s2c/msg
指令: {"Servo": 4} 抬杆（180°）
      {"Servo": 0} 落杆（0°）
"""
import argparse
import json
import time

import paho.mqtt.client as mqtt

from config import settings
from log_utils import get_logger

logger = get_logger("gate")


class GateController:
    def __init__(self):
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"gate_control_{int(time.time())}",
        )
        if settings.MQTT_USERNAME and settings.MQTT_PASSWORD:
            self.client.username_pw_set(
                settings.MQTT_USERNAME, settings.MQTT_PASSWORD)
        self.topic = settings.MQTT_PUB_TOPIC

    def connect(self):
        self.client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, 5)
        self.client.loop_start()
        time.sleep(0.3)  # 等连接建立

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, payload: dict) -> bool:
        msg = json.dumps(payload, ensure_ascii=False)
        result = self.client.publish(self.topic, msg)
        result.wait_for_publish(timeout=5)
        ok = result.rc == mqtt.MQTT_ERR_SUCCESS
        status = "OK" if ok else f"rc={result.rc}"
        logger.info(f"[MQTT] -> {msg}   [{status}]")
        return ok

    def open(self):
        logger.info("Gate OPEN")
        return self.publish(settings.SERVO_OPEN)

    def close(self):
        logger.info("Gate CLOSE")
        return self.publish(settings.SERVO_CLOSE)


def open_gate():
    """供 test_parking.py 调用的便捷函数。"""
    g = GateController()
    g.connect()
    try:
        return g.open()
    finally:
        g.disconnect()


def close_gate():
    g = GateController()
    g.connect()
    try:
        return g.close()
    finally:
        g.disconnect()


def main():
    parser = argparse.ArgumentParser(description="闸机控制")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--open", action="store_true", help="抬杆 Servo=4")
    group.add_argument("--close", action="store_true", help="落杆 Servo=0")
    args = parser.parse_args()

    g = GateController()
    g.connect()
    try:
        g.open() if args.open else g.close()
    finally:
        g.disconnect()


if __name__ == "__main__":
    main()
