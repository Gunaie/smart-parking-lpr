# 智能社区车牌识别系统 — 项目记忆

## 项目起源

2026-10-01 启动。用户持有了一份约 1740 行的教程（`1.txt`），目标是按照教程完成"智能社区车牌识别系统"，但环境严重受限：只有一台 Windows 笔记本（自带摄像头，无 GPU、无 USB 摄像头、无 AIoT 实训箱、无舵机、无扬声器外设）。用户要求先充分调研 → 确定开发计划 → 批准后分阶段执行（每阶段完成需用户确认）。

## 发展历程

### 阶段 0：环境调研与搭建
- **发现无 conda**：遍历常见安装路径均不存在，放弃 conda 方案
- **转向 uv**：本机已装 uv 0.11.5 且已托管 Python，用户事先授权"都行，你决定"
- **创建 .venv**：Python 3.11.15，清华源安装 76 个依赖包
- **hyperlpr3 Windows 兼容修复**：模型目录 HOMEPATH → USERPROFILE；zip 句柄未关闭 → with 语句
- **网络修复**：GitHub Release 直连被重置 → gh-proxy.com 镜像下载 yolo11n.pt

### 阶段 1：两阶段识别流水线
- 设计 `PlateRecognizer` 类：detect → recognize → draw_detections → save_results → __call__
- YOLO 检测用 `best.pt`，无模型时用 HyperLPR3 内置检测框兜底
- **裁剪策略**：15% padding 外扩 + 边界保护（max/min），numpy 切片 image[y1:y2, x1:x2]
- **二级兜底**：矩形裁剪 OCR 失败时，按 IoU>0.3 匹配整帧 LPR 结果（四点透视矫正）
- 中文车牌用 PIL + 微软雅黑绘制
- 实测：沪AD07979（99.89%）、皖AD05119（94.95%）、极端倾斜图走兜底（73.12%）

### 阶段 2：数据集获取与训练
- **关键发现**：Ultralytics Platform 公开数据集图片列表接口匿名可访问
- **免登录重建**：分页拉取 1875 张图片 + bbox 标注 → 下载 → 生成 dataset.yaml，全程零登录
- **训练**：CPU 上 yolo11n.pt，10 epochs 后提前收尾（第 8 epoch 最佳：P 0.997 / R 0.988 / mAP50 0.995 / mAP50-95 0.723），约 65 分钟
- 修复：yolo11n.pt 下载失败、Arial.ttf 下载失败、runs 目录嵌套

### 阶段 3：闸机链路
- 安装 Mosquitto 2.0.20，配置匿名访问
- gate_control.py：paho 2.x VERSION2，{"Servo":4} 抬杆 / {"Servo":0} 落杆
- gate_simulator.py：tkinter 画布闸杆动画（0°↔180° 平滑过渡）
- test_parking.py：Excel In Lot 表两列 + _ensure_excel/_exists/_add/_delete
- 四场景全部跑通：正常进场写入+抬落杆、重复进场 REJECT 不抬杆、正常离场删记录+抬落杆、无记录离场 NO RECORD 不抬杆

### 阶段 4：TTS 语音播报
- edge-tts 在线主路径（微软 XiaoxiaoNeural）+ playsound3 播放
- 离线兜底：pywin32 SAPI 调用（无需额外下载模型）
- 按内容哈希缓存 mp3
- 提供 welcome(plate, space)、goodbye(plate)、speak(text)

### 阶段 5：FastAPI + Vue3 综合系统
- server.py：/state /entry(422/409) /exit(404) /reset /video(MJPEG) /snapshot /stream(SSE 含 ping)
- static/index.html：Vue3 CDN 单文件，CSS Grid 4×6 车位，EventSource 实时推送
- **Excel 三列兼容**：test_parking.py 用两列（模块五），server.py 加 Space 列（模块六），_load_sheet 自动补列头
- **解决的坑**：8000/8001 被 Docker/WSL 占 → 改用 8765；PlateRecognizer 只接受路径 → 写临时文件；SSE data 字段单引号 → json.dumps()

### 阶段 6：端到端验收
- API 全量测试通过：/state /entry /exit /reset /snapshot /docs /stream
- 浏览器验证：Vue3 页面加载、4×6 车位网格、SSE 连接、进场/离场动画、统计数字更新

## 关键决策记录

| 时间 | 决策 | 理由 |
|------|------|------|
| 阶段0 | 用 uv 代替 conda | 本机无 conda，用户已授权 |
| 阶段0 | 清华源安装依赖 | 用户偏好 |
| 阶段2 | 匿名拉取平台数据集 | 用户不愿注册账号，无需登录 |
| 阶段2 | 10 epoch 提前收尾 | 第 8 epoch 已达 mAP50 0.995，继续训练边际收益低 |
| 阶段2 | yolo11n 而非 yolov13 | 官方说 v13 更大更慢，n 系列足够 |
| 阶段4 | edge-tts + SAPI 双路径 | 在线效果好，离线兜底保证无外网可用 |
| 阶段5 | Excel 兼容三列 | test_parking.py 和 server.py 共用，不破坏阶段3 |
| 阶段5 | SSE 而非轮询 | 教程推荐，减少无效请求 |

## 项目产出

- Python 文件：10 个
- 训练模型：1 个（21.2MB）
- 数据集：1875 张（212MB）
- 训练产物：runs/detect/train/（最佳第8 epoch）

## 持续运行状态

- Mosquitto MQTT Broker：localhost:1883
- FastAPI 服务：http://localhost:8765
- 闸机模拟器：tkinter 窗口
- YOLO 模型：已加载

## 知识沉淀

1. **Ultralytics Platform 匿名 API**：公开数据集的 images/list API 无需认证即可获取完整元数据和 CDN 图片 URL，适合不方便注册的场景。
2. **paho 2.x API 迁移**：`mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)` 与 1.x 不兼容，必须指定 callback_api_version。
3. **sse-starlette 事件格式**：`EventSourceResponse` 的 `data` 字段必须是字符串，dict 会被序列化为单引号导致前端 JSON.parse 失败。
4. **Windows 字体文件沙箱**：Ultralytics 训练时尝试下载 Arial.ttf 到用户目录，沙箱环境可能阻止此写入，需预先放置字体文件或设 `plots=False`。
5. **端口冲突排查**：Windows 下 Docker/WSL 常占用 8000/8001，netstat -ano 可定位进程。
