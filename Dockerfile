# Lightweight demo of the edge-inference service.
#
# Build weights first (python main.py train && python main.py quantize), then:
#   docker build -t resnet-edge-eval --build-arg MODEL=models/resnet18w05_int8_ptq.pt .
#   docker run -p 8000:8000 resnet-edge-eval
#
# Targets x86_64 Linux by default (python:3.11-slim + CPU-only PyTorch wheels).
# For ARM devices (Jetson / Raspberry Pi 4/5, aarch64), build on an arm64
# host/runner or swap the base image for a ready-made arm64 PyTorch image:
#   FROM pytorch/pytorch:2.13.0-cpu
# (onnxruntime installs a native aarch64 wheel automatically on arm64 hosts.)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch wheels keep the image several hundred MB smaller. torch is
# installed first so its CPU build is not shadowed by the generic index.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 torchvision==0.28.0 \
    && pip install onnxruntime==1.30.0 fastapi==0.136.1 uvicorn==0.52.4 \
        python-multipart==0.0.28 pillow==12.3.0 numpy==1.26.4

COPY main.py ./
COPY src ./src

# The quantized model artifact (must exist before building the image).
ARG MODEL=models/resnet18w05_int8_ptq.pt
COPY ${MODEL} /app/models/resnet18w05_int8_ptq.pt
RUN mkdir -p /app/results /app/data

EXPOSE 8000
CMD ["uvicorn", "src.serve:make_live_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]