FROM python:3.11-slim

# 换用清华 Debian 镜像源（加速国内构建）
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

WORKDIR /app

# OpenCV headless 仍需最小 GUI 库
# 注：如需在结果图中正确显示中文车牌，可手动添加 fonts-noto-cjk（下载较慢，可选）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 预下载 HyperLPR3 模型（避免首次运行时下载）
RUN python -c "from hyperlpr3 import LicensePlateCatcher; LicensePlateCatcher(detect_level=1)" \
    || echo "HyperLPR3 模型预下载失败，将在首次运行时下载"

COPY . .

RUN mkdir -p data/tts_cache uploads outputs

EXPOSE 8765

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8765"]
