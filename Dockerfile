
# Prebuilt training image with TF 2.15 + Py3.10
FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-15.py310:latest

ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=1
WORKDIR /trainer

# Only small libs you need beyond the base image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_furniture_tpu.py /trainer/train_furniture_tpu.py

# Make sure the path to python file is correct below.
ENTRYPOINT ["python", "-u", "/trainer/train_furniture_tpu.py"]    













# ########## ALTERNATIVELY - FOR ALL CPU BASED TRAINING USE THESE LINES BELOW #############
# # Lightweight base with Python 3.10
# FROM python:3.10-slim

# # Make logs flush immediately & quiet some TF spam
# ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=1

# # System deps (libgl1 is handy when Pillow/OpenCV decode images)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1 ca-certificates curl && \
#     rm -rf /var/lib/apt/lists/*

# # Create an unprivileged user for security
# RUN useradd -m trainer
# WORKDIR /trainer

# # ---- Python deps ----
# # If you have a requirements.txt, copy and install it (recommended).
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # ---- Your training code ----
# # Copy your training script into the image.
# # (Adjust the filename if yours is different.)
# COPY train_furniture_tpu.py /trainer/train_furniture_tpu.py

# # If you have any helper modules, copy them too:
# # COPY utils.py /trainer/utils.py
# # COPY src/ /trainer/src/

# # Drop privileges
# USER trainer

# # Vertex will append your CLI args after this entrypoint.
# ENTRYPOINT ["python", "-u", "/root/train_furniture_tpu.py"]


#################### ALTERNATIVELY IF YOU ARE USING A TPU #######################
# TPU pod base; Google recommends this for TF+TPU training containers
# (Python 3.10 variant)
#FROM us-docker.pkg.dev/vertex-ai/training/tf-tpu-pod-base-cp310:latest
# Install TensorFlow 2.15 wheel compatible with TPU VMs + libtpu
# (docs show installing TF wheel & libtpu explicitly)
# https://cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm
#RUN pip install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.15.0/tensorflow-2.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.9.0/libtpu.so -o /lib/libtpu.so

