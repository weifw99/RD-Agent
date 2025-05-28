# 使用 ARM64 兼容的基础镜像
FROM --platform=linux/arm64 python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    vim \
    git  \
    libomp5  \
    libomp-dev  \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置 python 和 pip 命令别名
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

# 安装 PyTorch 2.2.1 CPU 版本
RUN pip install --upgrade pip && \
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1

# 设置工作目录
WORKDIR /workspace

# 安装 Python 依赖（保持原始版本）
RUN pip install numpy && \
    pip install pandas && \
    pip install torch_geometric && \
    pip install pytorch_lightning && \
    pip install ogb && \
    pip install networkx && \
    pip install scikit-learn && \
    pip install catboost && \
    pip install xgboost && \
    pip install sparse && \
    pip install lightgbm==3.3.5 && \
    pip install pyarrow && \
    pip install fastparquet && \
    pip install optuna