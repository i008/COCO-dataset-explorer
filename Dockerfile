FROM continuumio/anaconda3

RUN apt update && apt install -y build-essential
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN pip install cython --no-cache-dir
RUN mkdir /cocodemo
ADD environment.yml requirements.txt coco_explorer.py cocoinspector.py config.py pycoco.py vis.py /cocodemo/
WORKDIR /cocodemo
RUN conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["/opt/conda/bin/streamlit", "run", "coco_explorer.py", "--"]
