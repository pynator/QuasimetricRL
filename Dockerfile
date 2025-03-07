FROM nvcr.io/nvidia/pytorch:25.02-py3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg cmake swig

RUN python -m pip install --upgrade pip
RUN pip install mujoco==3.1.6
RUN pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git
RUN pip install torch-tb-profiler
RUN pip install moviepy
RUN pip install wandb