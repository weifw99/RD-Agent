FROM --platform=linux/arm64 python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    libomp5  \
    libomp-dev  \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置 python 命令别名
RUN ln -s /usr/bin/python3 /usr/bin/python

# 安装 PyTorch 2.2.1（CPU）及相关依赖
RUN python -m pip install --upgrade pip && \
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install numpy && \
    pip install --upgrade cython && \
    pip install pandas && \
    pip install torch_geometric && \
    pip install ogb && \
    pip install networkx && \
    pip install scikit-learn && \
    pip install catboost && \
    pip install xgboost && \
    pip install sparse

# 工作目录
WORKDIR /workspace

CMD ["python"]