FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# For the convenience for users in China mainland
COPY docker/apt-sources.list /etc/apt/sources.list

# Install some basic utilities
RUN rm /etc/apt/sources.list.d/nvidia-ml.list \
 && rm /etc/apt/sources.list.d/cuda.list \
 && apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    g++ \
    libusb-1.0-0 \
    cmake \
    libssl-dev \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.3 \
 && conda clean -ya
COPY --chown=user docker/.condarc /home/user/.condarc

# CUDA 11.1-specific steps
RUN conda install -y -c conda-forge cudatoolkit=11.1.1 \
 && conda install -y -c pytorch \
    "pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0" \
    "torchvision=0.9.1=py38_cu111" \
 && conda clean -ya

# Alter sources for the convenience of users located in China mainland.
RUN pip config set global.index-url https://pypi.douban.com/simple
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV CUDA_HOME=/usr/local/cuda
RUN bash -c "git clone --recursive https://github.com/traveller59/spconv.git"
# We manually download and install cmake since the requirements of spconv is newer than
# that included in apt for ubuntu18.
RUN curl -sLo cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v3.20.1/cmake-3.20.1.tar.gz \
 && tar -xvf cmake.tar.gz \
 && cd cmake-3.20.1 \
 && ./configure \
 && make -j4 && sudo make install
RUN sudo apt-get update && sudo apt-get install -y libboost-dev \
 && sudo rm -rf /var/lib/apt/lists/*
COPY docker/spconv.sh spconv.sh
RUN bash spconv.sh

CMD ["python3"]
