# 智能社区车牌识别系统 — 交接文档

> 最后更新：2026-10-01

## 一、项目简介

基于教程（`1.txt`）实现的端到端车牌识别停车管理系统，在纯 Windows 笔记本（无 GPU、无 USB 摄像头、无 AIoT 硬件）环境下完成全部功能。

**核心链路**：图片/摄像头输入 → YOLO11n 车牌检测 → HyperLPR3 OCR 识别 → 业务判断（查重/查记录/车位分配）→ Excel 记录 → MQTT 闸机联动 → TTS 语音播报 → SSE 实时推送前端。

## 二、环境要求

| 项目 | 说明 |
|------|------|
| Python | 3.11.15，虚拟环境在 `.venv/`（uv 创建） |
| 运行命令 | 一律使用 `.venv\Scripts\python.exe`（系统 PATH 无 python） |
| 依赖 | `requirements.txt`，清华源安装 |
| MQTT Broker | Mosquitto 2.0.20（已装 `C:\Program Files\mosquitto`），配置文件 `mosquitto_local.conf`（匿名、监听 1883） |
| 端口 | 服务默认 **8765**（8000/8001 被 Docker/WSL 占用） |

**启动顺序：**
```powershell
# 1. MQTT Broker（如未运行）
& "C:\Program Files\mosquitto\mosquitto.exe" -c e:\Trae\MHYS\mosquitto_local.conf

# 2. 闸机模拟器（可选，观察抬落杆动画）
.\.venv\Scripts\pythonw.exe gate_simulator.py

# 3. 主服务（--no-gate 跳过闸机，--no-tts 跳过语音）
.\.venv\Scripts\python.exe server.py --port 8765
```

**访问入口：**
- 系统首页：http://localhost:8765
- API 文档：http://localhost:8765/docs
- 车位状态：http://localhost:8765/state

## 三、文件结构与职责

```
e:\Trae\MHYS\
├── config.py            # 全局配置（路径/MQTT/TTS/车位，禁止业务代码写死路径）
├── plate_pipeline.py    # PlateRecognizer：YOLO 检测 + 15% padding 裁剪 + HyperLPR3 OCR
├── train.py             # YOLO11n 训练脚本（CPU 可跑，产出复制到 models/best.pt）
├── gate_control.py      # GateController：MQTT 发 {"Servo":4} 抬杆 / {"Servo":0} 落杆
├── gate_simulator.py    # tkinter 闸杆动画（无舵机时观察指令）
├── test_parking.py      # 命令行进出场测试（--entry/--exit）
├── tts_utils.py         # edge-tts 在线播报 + SAPI 离线兜底，welcome/goodbye/speak
├── server.py            # FastAPI 后端（/state /entry /exit /reset /video /snapshot /stream）
├── static/index.html    # Vue3 前端单文件（4×6 车位网格、SSE、上传/拍照）
├── mosquitto_local.conf # MQTT 本地配置
├── models/best.pt       # 训练产出（mAP50=0.995, mAP50-95=0.723）
├── dataset/             # CCPD 车牌数据集 1875 张 + dataset.yaml
├── samples/             # 测试图片 13 张
├── data/parking_record.xlsx  # 停车记录（In Lot 表：车牌/入场时间/车位号）
├── uploads/ outputs/    # 运行时上传与识别输出
└── runs/detect/train/   # 训练产物归档
```

## 四、关键技术决策

1. **数据集免登录获取**：Ultralytics Platform 公开数据集接口匿名可访问，逐页拉取 1875 张图片 + 标注重建 YOLO 数据集（train 1555 / val 164 / test 156）。
2. **训练提前收尾**：原计划 30 epochs，第 8 epoch 达最佳（mAP50 0.995），10 epoch 后停止（约 65 分钟）。
3. **双检测兜底**：YOLO 无结果时自动退回 HyperLPR3 内置检测框；裁剪 OCR 失败时按 IoU>0.3 匹配整帧 LPR 结果（四点透视矫正，处理倾斜车牌）。
4. **TTS 双路径**：edge-tts 在线（微软 Xiaoxiao）+ pywin32 SAPI 离线兜底；按内容哈希缓存 mp3。
5. **Excel 三列兼容**：test_parking.py 用两列（教程模块五），server.py 用三列加车位号（模块六），`_load_sheet` 自动补列头。

## 五、已解决的坑（勿再踩）

| 问题 | 原因 | 解决 |
|------|------|------|
| hyperlpr3 Windows 崩溃 | 模型目录误用 HOMEPATH（无盘符）；zip 句柄未关就删文件 | 已修，模型在 `C:\Users\gunaie\.hyperlpr3\` |
| yolo11n.pt 下载失败 | GitHub 直连被重置 | gh-proxy.com 镜像，文件在项目根目录 |
| Arial.ttf 下载失败 | 沙箱禁写 Ultralytics 目录 | 手动放 `C:\Users\gunaie\AppData\Roaming\Ultralytics\Arial.ttf`；训练加 `plots=False` |
| runs 目录嵌套 | 显式传 project 与新 ultralytics 的 runs_dir 叠加 | train.py 不传 project |
| 8000/8001 端口被占 | Docker/WSL | 改用 8765 |
| /entry 500 | PlateRecognizer 只接受文件路径，传了 numpy | 写临时文件再调用 |
| SSE 前端 JSON 解析失败 | data 字段是 dict 被序列化成单引号 | data 字段统一 `json.dumps()` 成字符串 |
| Excel 写入失败 | WPS/Excel 占用文件 | 测试前关闭 |

## 六、测试验证记录

- 四场景命令行测试：正常进场 / 重复进场(409 REJECT) / 正常离场 / 无记录离场(404 NO RECORD) 全部通过
- API 验收：/state /entry /exit /reset /snapshot /docs /stream 全部通过（记录见 `outputs/acceptance.txt`）
- 识别实测：沪AD07979 99.80%、皖AD05119 94.95%、皖AF06700 99.78%、极端倾斜图走兜底链路 73.12%
- 已知限制：320×320 监控级小图车牌过小会漏检（超出入口近景场景）

## 七、后续维护要点

- 新增依赖后更新 `requirements.txt`
- 重新训练用 `python train.py --epochs 30 --batch 8`，自动复制到 models/best.pt
- 换摄像头改 `config.py` 的 `CAMERA_INDEX`
- 接真实舵机：修改 `MQTT_BROKER` 指向实训箱即可，协议不变
