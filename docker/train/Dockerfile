FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive 

# container marker
RUN echo "PS1='🐳  \[\033[1;36m\]\h \[\033[1;34m\]\W\[\033[0;35m\] \[\033[1;36m\]# \[\033[0m\]'" >> /root/.bashrc

# usefull
RUN apt-get update && apt-get upgrade && apt-get install -y python3.10
RUN apt install python3-pip -y
RUN apt-get install zip -y
RUN apt-get install nano -y

# install ultralytics
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install ultralytics

# setup workspace
WORKDIR /root
RUN mkdir -p src

COPY ../config/train/ /root/config/
COPY ../src/train_YOLOv10/ /root/src/
WORKDIR /root/src/