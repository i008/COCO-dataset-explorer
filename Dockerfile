FROM continuumio/miniconda3

RUN apt update && apt install -y build-essential
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install cython --no-cache-dir

RUN mkdir /cocodemo
ADD . /cocodemo
WORKDIR /cocodemo

RUN pip install -r requirements.txt
RUN git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI & pip install easyimages==1.1