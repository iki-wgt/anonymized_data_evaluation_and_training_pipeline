FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive 

# container marker
RUN echo "PS1='🐳  \[\033[1;36m\]\h \[\033[1;34m\]\W\[\033[0;35m\] \[\033[1;36m\]# \[\033[0m\]'" >> /root/.bashrc

# usefull
RUN apt-get update && apt-get upgrade -y
RUN apt install python3-pip -y
RUN apt-get install zip -y
RUN apt-get install nano -y

# install ultralytics
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install ultralytics

# install enhanced COCO API - needs some changed files because source is not for python3
RUN pip3 uninstall pycocotools -y # remove old install if there
RUN pip3 install cython
RUN apt-get install gcc
RUN apt-get -y install git
WORKDIR /root/
RUN git clone https://github.com/yhsmiley/fdet-api.git
COPY ../cocoapi/ /root/fdet-api/
WORKDIR /root/fdet-api/PythonAPI
RUN python3 setup.py install
RUN make

# easy and simple boxes
RUN pip3 install bounding-box
# excel file write
RUN pip3 install openpyxl
# image analyze
RUN pip3 install scikit-image
RUN pip3 install pandas openpyxl


# setup workspace
WORKDIR /root
RUN mkdir -p src

COPY ../config/eval/ /root/config/
COPY ../src/eval/ /root/src/
WORKDIR /root/src/