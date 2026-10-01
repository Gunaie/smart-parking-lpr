# -*- coding: utf-8 -*-
"""闸机模拟器：订阅 MQTT 指令，用 tkinter 画布展示闸杆抬/落动画。

本机无舵机时，用它直观观察 gate_control.py / test_parking.py 是否正确发出指令。
运行后会弹出一个窗口，显示当前闸杆角度（0° 落杆 / 180° 抬杆）。
"""
import json
import tkinter as tk
from threading import Thread

import paho.mqtt.client as mqtt

from config import settings

SERVO_TO_ANGLE = {0: 0, 1: 45, 2: 90, 3: 135, 4: 180}


class GateSimulator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("闸机模拟器")
        self.root.geometry("420x300")

        self.angle = 0  # 0 = 落杆，180 = 抬杆
        self.target_angle = 0
        self.status = tk.StringVar(value="等待指令...")

        tk.Label(self.root, text="闸机模拟器", font=("微软雅黑", 14, "bold")).pack(pady=6)
        tk.Label(self.root, textvariable=self.status, fg="#444").pack()

        self.canvas = tk.Canvas(self.root, width=360, height=200, bg="#e8e8e8")
        self.canvas.pack(pady=6)

        # 立柱和底座
        self.canvas.create_rectangle(170, 160, 200, 190, fill="#666")
        self.canvas.create_rectangle(120, 188, 250, 200, fill="#444")
        # 闸杆（从立柱顶部向右伸出），用一条粗线
        self.bar = self.canvas.create_line(185, 160, 185, 60,
                                           width=10, fill="#e74c3c", capstyle=tk.ROUND)

        self._mqtt_connect()
        self.root.after(50, self._animate)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _mqtt_connect(self):
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id="gate_simulator")
        if settings.MQTT_USERNAME and settings.MQTT_PASSWORD:
            self.client.username_pw_set(
                settings.MQTT_USERNAME, settings.MQTT_PASSWORD)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = lambda *a: self.status.set("MQTT 已断开")

        def run():
            try:
                self.client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, 5)
                self.client.loop_forever()
            except Exception as e:
                self.status.set(f"连接失败: {e}")

        Thread(target=run, daemon=True).start()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(settings.MQTT_PUB_TOPIC)
        self.status.set(
            f"已连接 {settings.MQTT_BROKER}:{settings.MQTT_PORT}  "
            f"订阅 {settings.MQTT_PUB_TOPIC}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            servo = payload.get("Servo")
            if servo in SERVO_TO_ANGLE:
                self.target_angle = SERVO_TO_ANGLE[servo]
                action = "抬杆" if servo == 4 else ("落杆" if servo == 0 else f"角度{servo}")
                self.status.set(f"收到指令 Servo={servo} -> {action}")
        except Exception as e:
            self.status.set(f"消息解析失败: {e}")

    def _animate(self):
        if self.angle < self.target_angle:
            self.angle = min(self.angle + 3, self.target_angle)
        elif self.angle > self.target_angle:
            self.angle = max(self.angle - 3, self.target_angle)

        # 闸杆起点(185,160)，角度从0(竖直下落)到180(水平向右)
        # 用 angle 的补角计算端点：angle=0 -> 竖直向下(185,160+100)
        # angle=180 -> 水平向右(185+100, 160)
        import math
        rad = math.radians(self.angle)
        length = 100
        ex = 185 + length * math.sin(rad)
        ey = 160 + length * math.cos(rad)
        self.canvas.coords(self.bar, 185, 160, ex, ey)
        self.root.after(20, self._animate)

    def _on_close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    GateSimulator().run()
