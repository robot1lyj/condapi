FROM nvcr.io/nvidia/pytorch:26.05-py3
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /opt/openwam
COPY upstream/ /opt/openwam/
# Preserve NVIDIA's Thor-capable torch/vision/triton; never install upstream cu128 wheels.
RUN python -m pip freeze | grep -Ei '^(torch|torchvision|torchaudio|triton|numpy)==' > /tmp/nvidia-constraints.txt && \
    python -m pip install --constraint /tmp/nvidia-constraints.txt \
      'transformers>=5.5,<6' 'diffusers>=0.38,<1' 'accelerate>=1,<2' \
      'hydra-core>=1.3,<2' 'omegaconf>=2.3,<3' 'safetensors>=0.4,<1' \
      'sentencepiece>=0.1.99,<1' 'ftfy>=6.1,<7' 'einops>=0.7,<1' \
      'websockets>=16,<17' 'msgpack>=1,<2' 'imageio[ffmpeg]>=2.28,<3' \
      'opencv-python-headless>=4.7,<5' 'timm>=1.0.20,<1.1' 'modelscope>=1.13,<2' && \
    python -m pip install --no-deps -e . && \
    python -m pip freeze > /opt/openwam/thor-environment.txt
RUN python -m pip install --constraint /tmp/nvidia-constraints.txt \
      'h5py>=3.8,<4' 'pyarrow>=14,<26' 'pandas>=2,<4' 'scipy>=1.10,<2' 'av>=12,<18' && \
    python -c "from openwam.dataloader.transforms.normalize import Normalizer" && \
    python -m pip freeze > /opt/openwam/thor-environment.txt
ENV PYTHONPATH=/opt/openwam
CMD ["python", "-m", "openwam.deploy.server", "--help"]
