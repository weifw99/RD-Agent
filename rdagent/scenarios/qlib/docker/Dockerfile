FROM --platform=linux/arm64 python:3.10-slim

RUN apt-get update && apt-get install -y  \
    curl  \
    vim  \
    git  \
    libomp5  \
    libomp-dev  \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置 python/pip 命令别名
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# 添加 libomp

# 安装 PyTorch 2.2.1（CPU 版）
RUN pip install --upgrade pip && \
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# 克隆 Qlib 并固定提交版本
RUN git clone https://github.com/microsoft/qlib.git /workspace/qlib

WORKDIR /workspace/qlib

# RUN git fetch && git reset v0.9.6 --hard
RUN git fetch && git reset df557d29d516f6741fd8b3b26cea05f8516c8435 --hard


# 安装 Python 依赖（保留版本）
RUN pip install --upgrade cython && \
    pip install -e . && \
    pip install catboost && \
    pip install xgboost && \
    pip install lightgbm && \
    pip install scipy==1.11.4 && \
    pip install tables && \
    pip install joblib==1.4.2