# syntax=docker/dockerfile:1.7
# Base images: runtime + ONNX Runtime only. The Argos binaries live in the
# app image built from app.Dockerfile with BASE_IMAGE pointing here.
#
# Build examples:
#   docker build -f docker/base.Dockerfile --target base-cpu   -t argos-ort:cpu-1.31 .
#   docker build -f docker/base.Dockerfile --target base-gpu   -t argos-ort:gpu-1.31 .
#   docker build -f docker/base.Dockerfile --target base-intel -t argos-ort:intel-1.31 .
#   docker build -f docker/base.Dockerfile --target base-amd   -t argos-ort:amd-1.31 .

ARG ORT_VERSION=1.31.0
ARG TARGETARCH=amd64

# --- CPU: official ORT prebuilt --------------------------------------------
FROM debian:13-slim AS base-cpu

ARG ORT_VERSION
ARG TARGETARCH
ARG GIT_SHA=dev
LABEL org.opencontainers.image.revision=$GIT_SHA

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl poppler-utils tini libstdc++6 libgomp1 passwd \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then arch="aarch64"; else arch="x64"; fi; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${arch}-${ORT_VERSION}.tgz" -o /tmp/ort.tgz; \
    tar -xzf /tmp/ort.tgz -C /opt; \
    mv "/opt/onnxruntime-linux-${arch}-${ORT_VERSION}" /opt/onnxruntime; \
    cp /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/; \
    ldconfig

ENV ARGOS_ORT_LIB=/usr/local/lib/libonnxruntime.so \
    HOME=/home/argos \
    ARGOS_HOME=/home/argos/.local/share/argos

RUN userdel -r ubuntu 2>/dev/null || true; \
    if ! id -u argos >/dev/null 2>&1; then useradd -m -u 1000 argos; fi
USER argos
WORKDIR /home/argos
VOLUME ["/home/argos"]
EXPOSE 7712 7711
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["argosd", "--addr", "0.0.0.0:7712"]

# --- GPU: official ORT CUDA prebuilt + CUDA runtime ------------------------
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base-gpu

ARG ORT_VERSION
ARG GIT_SHA=dev
LABEL org.opencontainers.image.revision=$GIT_SHA

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl poppler-utils tini libstdc++6 libgomp1 passwd \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}.tgz" -o /tmp/ort.tgz; \
    tar -xzf /tmp/ort.tgz -C /opt; \
    mv "/opt/onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}" /opt/onnxruntime; \
    cp /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/; \
    ldconfig

ENV ARGOS_ORT_LIB=/usr/local/lib/libonnxruntime.so \
    HOME=/home/argos \
    ARGOS_HOME=/home/argos/.local/share/argos
RUN userdel -r ubuntu 2>/dev/null || true; \
    if ! id -u argos >/dev/null 2>&1; then useradd -m -u 1000 argos; fi
USER argos
WORKDIR /home/argos
VOLUME ["/home/argos"]
EXPOSE 7712 7711
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["argosd", "--addr", "0.0.0.0:7712"]

# --- Intel: source-built ORT with OpenVINO, runtime base -------------------
FROM openvino/ubuntu24_dev:2026.4.1 AS ort-intel

ARG ORT_VERSION
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential python3 \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "v${ORT_VERSION}" --recurse-submodules \
      --shallow-submodules https://github.com/microsoft/onnxruntime.git /ort
WORKDIR /ort
RUN ./build.sh --config Release --build_shared_lib --parallel --skip_tests \
      --skip_submodule_sync --use_openvino CPU \
      --compile_no_warning_as_error \
      --allow_running_as_root --build_dir /build

FROM openvino/ubuntu24_runtime:2026.4.1 AS base-intel

ARG GIT_SHA=dev
LABEL org.opencontainers.image.revision=$GIT_SHA
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl poppler-utils tini libstdc++6 libgomp1 passwd \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ort-intel /build/Release/libonnxruntime.so* /usr/local/lib/
COPY --from=ort-intel /build/Release/libonnxruntime_providers_*.so /usr/local/lib/
RUN ldconfig
ENV ARGOS_ORT_LIB=/usr/local/lib/libonnxruntime.so \
    HOME=/home/argos \
    ARGOS_HOME=/home/argos/.local/share/argos
RUN userdel -r ubuntu 2>/dev/null || true; \
    if ! id -u argos >/dev/null 2>&1; then useradd -m -u 1000 argos; fi
USER argos
WORKDIR /home/argos
VOLUME ["/home/argos"]
EXPOSE 7712 7711
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["argosd", "--addr", "0.0.0.0:7712"]

# --- AMD: source-built ORT with MIGraphX, runtime base ---------------------
FROM rocm/dev-ubuntu-24.04:6.4 AS ort-amd

ARG ORT_VERSION
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential python3 \
      migraphx-dev miopen-hip-dev rocblas-dev hipblaslt-dev half \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/rocm/include/migraphx \
    && ln -sf /opt/rocm/lib/migraphx/include/migraphx/version.h /opt/rocm/include/migraphx/version.h
RUN git clone --depth 1 --branch "v${ORT_VERSION}" --recurse-submodules \
      --shallow-submodules https://github.com/microsoft/onnxruntime.git /ort
WORKDIR /ort
RUN ./build.sh --config Release --build_shared_lib --parallel --skip_tests \
      --skip_submodule_sync --use_migraphx --migraphx_home /opt/rocm \
      --compile_no_warning_as_error \
      --allow_running_as_root --build_dir /build

FROM rocm/dev-ubuntu-24.04:6.4 AS base-amd

ARG GIT_SHA=dev
LABEL org.opencontainers.image.revision=$GIT_SHA
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl poppler-utils tini libstdc++6 libgomp1 passwd \
      migraphx miopen-hip half \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ort-amd /build/Release/libonnxruntime.so* /usr/local/lib/
COPY --from=ort-amd /build/Release/libonnxruntime_providers_*.so /usr/local/lib/
RUN ldconfig
ENV ARGOS_ORT_LIB=/usr/local/lib/libonnxruntime.so \
    HOME=/home/argos \
    ARGOS_HOME=/home/argos/.local/share/argos
RUN userdel -r ubuntu 2>/dev/null || true; \
    if ! id -u argos >/dev/null 2>&1; then useradd -m -u 1000 argos; fi
USER argos
WORKDIR /home/argos
VOLUME ["/home/argos"]
EXPOSE 7712 7711
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["argosd", "--addr", "0.0.0.0:7712"]
