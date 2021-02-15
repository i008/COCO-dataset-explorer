FROM continuumio/anaconda3

RUN apt update && apt install -y build-essential
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN pip install cython --no-cache-dir
RUN mkdir /cocodemo
ADD . /cocodemo
WORKDIR /cocodemo
RUN conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
RUN pip install -r requirements.txt
RUN git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI & pip install easyimages
