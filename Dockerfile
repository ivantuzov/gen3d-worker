# gen3d-worker: TRELLIS.2 image -> GLB для RunPod Serverless.
# Собирается GitHub Actions → ghcr.io/ivantuzov/gen3d-worker (публичный образ, как music-worker).
#
# Лицензии (всё допускает коммерческое использование):
#   TRELLIS.2 код+веса ............ MIT  (microsoft/TRELLIS.2, microsoft/TRELLIS.2-4B)
#   форк без nvdiffrast ........... MIT  (GaXxO-dev/TRELLIS.2-commercial-use, рендер на DRTK)
#   DRTK .......................... MIT  (facebookresearch/DRTK)
#   CuMesh / FlexGEMM / o-voxel ... MIT  (JeffreyXiang)
#   BiRefNet (снятие фона) ........ MIT  (ZhengPeng7/BiRefNet) вместо briaai/RMBG-2.0 (CC BY-NC!)
#   DINOv3 ViT-L .................. DINOv3 License (коммерция можно, gated: нужен одобренный доступ на HF)
#   НЕ брать: nvdiffrast/nvdiffrec (NVIDIA Source Code License — только research/evaluation)
#
# Сборка: docker buildx build --platform linux/amd64 -t <registry>/gen3d-worker:v1 .
#   Веса по умолчанию НЕ вшиваются (BAKE_WEIGHTS=0): внутри RunPod HF отдаёт быстрее, чем GHCR (~22 МБ/с),
#   и образ влезает в диск бегунка Actions. BAKE_WEIGHTS=1 — вшить TRELLIS.2-4B (~15 ГБ) + BiRefNet.
#   DINOv3 (1,2 ГБ, gated) качается при старте по HF_TOKEN — образ можно держать публичным.
#   Для приватного реестра можно вшить и DINOv3: --build-arg BAKE_DINO=1 --secret id=hf,env=HF_TOKEN
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1 \
    CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.6;8.9" MAX_JOBS=4 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip git ninja-build g++ \
      libgl1 libglib2.0-0 libjpeg-dev libwebp-dev ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja && \
    pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
      --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
      https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# CUDA-расширения собираем из исходников под sm_86/89 (A40/A6000=8.6, L40S=8.9)
RUN git clone --depth 1 https://github.com/facebookresearch/DRTK.git /tmp/drtk && \
    pip install --no-cache-dir --no-build-isolation /tmp/drtk && rm -rf /tmp/drtk
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh && \
    pip install --no-cache-dir --no-build-isolation /tmp/CuMesh && rm -rf /tmp/CuMesh
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM && \
    pip install --no-cache-dir --no-build-isolation /tmp/FlexGEMM && rm -rf /tmp/FlexGEMM
# Форк закрепить на конкретном коммите после первой удачной сборки (ARG TRELLIS_REF)
ARG TRELLIS_REF=main
RUN git clone --recursive https://github.com/GaXxO-dev/TRELLIS.2-commercial-use.git /app/TRELLIS.2 && \
    cd /app/TRELLIS.2 && git checkout $TRELLIS_REF && \
    pip install --no-cache-dir --no-build-isolation /app/TRELLIS.2/o-voxel
ENV PYTHONPATH=/app/TRELLIS.2

# Проверка, что nvdiffrast не затесался
RUN python -c "import importlib.util as u; assert u.find_spec('nvdiffrast') is None, 'nvdiffrast found'; assert u.find_spec('drtk') and u.find_spec('o_voxel'); print('ext ok')"

ARG BAKE_WEIGHTS=0
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python -c "from huggingface_hub import snapshot_download as s; \
    s('microsoft/TRELLIS.2-4B'); s('microsoft/TRELLIS-image-large', allow_patterns=['ckpts/ss_dec_conv3d_16l8_fp16*','pipeline.json']); \
    s('ZhengPeng7/BiRefNet')"; fi
ARG BAKE_DINO=0
RUN --mount=type=secret,id=hf,required=false \
    if [ "$BAKE_DINO" = "1" ]; then HF_TOKEN=$(cat /run/secrets/hf) python -c "from huggingface_hub import snapshot_download as s; \
    s('facebook/dinov3-vitl16-pretrain-lvd1689m')"; fi

COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
