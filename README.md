# 智能社区车牌识别系统

> 基于 YOLO11n + HyperLPR3 的端到端车牌检测识别与停车管理全栈应用。在 Windows CPU 笔记本上自训练模型、完成前后端开发与部署。

![CI](https://github.com/USERNAME/REPO/actions/workflows/ci.yml/badge.svg)

> 推送到 GitHub 后，将 `USERNAME/REPO` 替换为你的仓库地址即可显示 CI 状态。

---

## 架构概览

```
图像输入(上传/摄像头)
       ↓
YOLO11n 车牌检测 (自训练, mAP50=0.995)
       ↓
15% padding 裁剪 + HyperLPR3 OCR
       ↓
业务判断(查重/查记录/车位分配)
       ↓
Excel 记录 + MQTT 闸机联动 + TTS 语音播报 + SSE 前端推送
```

**技术栈**

| 层级 | 技术 |
|------|------|
| 检测 | YOLO11n (Ultralytics) |
| OCR | HyperLPR3 (ONNX) |
| 后端 | FastAPI + uvicorn + sse-starlette |
| 前端 | Vue3 (CDN) + CSS Grid |
| 实时推送 | SSE (Server-Sent Events) |
| 消息总线 | MQTT (Mosquitto) |
| 语音 | edge-tts + SAPI |
| 数据 | openpyxl |

---

## 快速启动

### 方式一：Docker（推荐，一键启动）

```bash
docker compose up -d
```

自动启动两个容器：

| 容器 | 服务 | 端口 |
|------|------|------|
| mhys-mqtt | Eclipse Mosquitto 2.x | 1883 |
| mhys-app | FastAPI 应用（已含模型与中文字体） | 8765 |

访问 http://localhost:8765

> 容器内限制（不影响核心功能）：
> - 无音频设备：TTS 生成 mp3 但不发声
> - 无摄像头：`/snapshot` `/video` 不可用，仅支持图片上传识别
> - 无中文字体：结果图中车牌文字可能显示为方块，不影响识别结果
> - 如需中文字体，在 Dockerfile 的 apt 安装中添加 `fonts-noto-cjk` 后重建镜像

### 方式二：本机运行（Windows，含摄像头与语音）

```powershell
# 1. 确保 MQTT Broker 运行（首次需安装 Mosquitto）
& "C:\Program Files\mosquitto\mosquitto.exe" -c mosquitto_local.conf

# 2. 启动闸机模拟器（可选，观察抬落杆动画）
.\.venv\Scripts\pythonw.exe gate_simulator.py

# 3. 启动主服务
.\.venv\Scripts\python.exe server.py --port 8765
```

访问 http://localhost:8765

---

## 核心功能

- **车牌检测**：自训练 YOLO11n 模型，CPU 单帧 98ms（模型热缓存后）
- **车牌识别**：HyperLPR3 OCR，支持多类型中国车牌
- **双兜底链路**：YOLO 漏检 → HyperLPR3 内置检测；裁剪 OCR 失败 → IoU 整帧匹配
- **车位管理**：24 车位 A1~D6，空闲自动分配
- **业务校验**：重复进场 409 / 车位已满 409 / 无记录离场 404 / 未检测到车牌 422
- **实时推送**：SSE 前端自动更新车位状态（无需轮询）
- **摄像头**：MJPEG 实时流 + 单帧截图 + 拍照识别
- **闸机仿真**：MQTT 协议抬落杆 + tkinter 动画窗口
- **语音播报**：edge-tts 在线 + SAPI 离线兜底，进场"欢迎光临"离场"一路顺风"

---

## 项目结构

```
├── config.py              # 全局配置（路径/MQTT/TTS/车位）
├── log_utils.py           # 统一日志（控制台+文件轮转+请求ID）
├── plate_pipeline.py      # PlateRecognizer：检测→裁剪→OCR→兜底
├── train.py               # YOLO11n 训练（CPU 可跑）
├── gate_control.py        # MQTT 闸机指令
├── gate_simulator.py      # tkinter 闸杆动画
├── test_parking.py        # 命令行进出场测试
├── tts_utils.py           # 双路径语音播报
├── server.py              # FastAPI 后端（8 个路由）
├── static/index.html      # Vue3 前端
├── tests/                 # pytest 测试套件
├── logs/                  # 运行日志（app.log，5MB×5 轮转）
├── models/best.pt         # 训练产出（20.3MB）
├── dataset/               # CCPD 1875 张 + dataset.yaml
└── samples/               # 测试图片
```

## 日志系统

统一使用 `logging` 模块（`log_utils.py`），替代 `print`：

- **双输出**：控制台 + 文件 `logs/app.log`（5MB × 5 轮转）
- **请求 ID 透传**：每个 HTTP 请求注入 `X-Request-ID`，跨模块（server→pipeline→gate→tts）日志带同一 ID，便于链路追踪
- **日志格式**：`时间 [级别] 模块 [请求ID]: 消息`
- **噪音抑制**：ultralytics / uvicorn.access 降级为 WARNING

```
2026-10-01 14:51:26 [INFO] server [entry001]: 进场: 沪AD07979 -> A1
2026-10-01 14:51:26 [INFO] gate   [entry001]: Gate OPEN
2026-10-01 14:51:26 [INFO] gate   [entry001]: [MQTT] -> {"Servo": 4}   [OK]
```

---

## 测试

```powershell
# 完整测试（需 best.pt + 服务运行）
.\.venv\Scripts\python.exe -m pytest tests/ -v

# CI 模式（跳过需模型/服务的测试，6 passed / 18 skipped）
.\.venv\Scripts\python.exe -m pytest tests/ -v -m "not requires_model and not requires_server"

# 仅识别流水线精度回归
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v

# 性能基准
.\.venv\Scripts\python.exe -m pytest tests/test_benchmark.py -v -s
```

**pytest markers**

| marker | 含义 |
|--------|------|
| `requires_model` | 需要 `models/best.pt`（CI 环境无模型时自动跳过） |
| `requires_server` | 需要服务运行在 localhost:8765 |

**CI/CD**：`.github/workflows/ci.yml` — push/PR 时自动安装依赖并运行 CI 模式测试。

**当前基准（CPU）**

```json
{
  "recognition_latency_s": {"avg": 0.098, "min": 0.085, "max": 0.107},
  "model_size_mb": 20.3,
  "api_entry_latency_s": 2.344
}
```

---

## 技术对比实验：YOLO11n vs HyperLPR3

完整报告见 [outputs/comparison_report.md](outputs/comparison_report.md)。

### 检测精度（test 集 156 张，IoU>0.5）

| 指标 | YOLO11n（自训练） | HyperLPR3（内置检测） |
|------|-------------------|----------------------|
| Precision | 0.9207 | **0.9934** |
| Recall | 0.9679 | 0.9679 |
| F1 | 0.9438 | **0.9805** |
| 检测延迟 | 165.0ms | **23.6ms** |

### 端到端识别（test 集 40 张，双方法交叉验证）

| 方案 | 检出率 | 一致率 | 延迟 |
|------|--------|--------|------|
| YOLO11n 检测 + 裁剪 + OCR | 95.0% | 86.8% | 245ms |
| HyperLPR3 pipeline（一体） | 95.0% | 86.8% | 48ms |

> 一致率：两种独立 OCR 方法结果完全相同的比例。剩余 13.2% 分歧全部为省份汉字混淆（粤/皖/京/宁），属于 OCR 模型固有局限。按 CCPD 数据集以安徽（皖）为主推断，真实整牌准确率约 **YOLO+OCR 92% / HyperLPR3 95%**。

### 选型结论

HyperLPR3 在**常规场景**下检测精度和速度均更优；项目采用 YOLO11n + OCR 解耦架构的理由：
- YOLO11n 可针对特定场景微调，HyperLPR3 检测器固定
- 检测与识别解耦，可独立优化
- 极端倾斜样本上 YOLO 检测框 + 整帧 OCR 兜底更鲁棒

---

## 训练

```powershell
.\.venv\Scripts\python.exe train.py --epochs 30 --batch 8
```

- 数据集：CCPD 车牌 1875 张（匿名从 Ultralytics Platform 重建）
- 最佳第 8 epoch：P=0.997 / R=0.988 / mAP50=0.995 / mAP50-95=0.723
- 训练时间：约 65 分钟（16 核 CPU）
- 产物自动复制到 `models/best.pt`

---

## 关键设计决策

1. **免登录数据集**：利用 Ultralytics Platform 公开接口匿名重建 CCPD 数据集，无需注册账号
2. **提前收尾**：第 8 epoch 已达 mAP50 0.995，10 epoch 后停止训练，避免边际收益浪费
3. **Excel 兼容**：server.py 三列（含车位号）与 test_parking.py 两列自动兼容
4. **SSE 实时推送**：减少轮询、即时反馈，比定时刷新更适合停车场即时场景

---

## 适用场景

- 智能社区/园区车辆进出管理
- 无 GPU 环境部署
- 作为 YOLO 目标检测 + OCR 识别 + Web 全栈的入门参考项目

## License

数据集 Apache-2.0，代码个人学习用途。
