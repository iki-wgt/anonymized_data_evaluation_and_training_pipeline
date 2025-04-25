# base
FROM python:3.9 as BASE

# container marker
RUN echo "PS1='🐳  \[\033[1;36m\]\h \[\033[1;34m\]\W\[\033[0;35m\] \[\033[1;36m\]# \[\033[0m\]'" >> /root/.bashrc

# Install base utilities
RUN apt update && apt install -y \
    python3-pip
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install packet manager anaconda for pytorch (recommended by DeepPrivacy2)
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_24.7.1-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path
ENV PATH=$CONDA_DIR/bin:$PATH

# Install DeepPrivacy2 Dependencies
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip uninstall -y numpy
RUN pip3 uninstall -y numpy
RUN pip3 install numpy==1.22.4
RUN pip3 install scipy==1.7.1

# Install DeepPrivacy2
WORKDIR /root/
RUN git clone https://github.com/hukkelas/deep_privacy2/
WORKDIR /root/deep_privacy2
RUN pip3 install -e .

# copy code
WORKDIR /root/
COPY ../src/anonymized_coco/ /root/deep_privacy2/

# install enhanced COCO API - needs some changed files because source is not for python3
RUN pip3 uninstall pycocotools -y # remove old install
RUN pip3 install cython
RUN apt-get install gcc
WORKDIR /root/
RUN git clone https://github.com/yhsmiley/fdet-api.git
COPY ../cocoapi/ /root/fdet-api/
WORKDIR /root/fdet-api/PythonAPI
RUN python3 setup.py install
RUN make

# go to path of execution
WORKDIR /root/deep_privacy2