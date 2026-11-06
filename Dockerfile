# syntax=docker/dockerfile:1.7

# --- build ------------------------------------------------------------------
FROM golang:1.27-bookworm AS build

ARG ORT_VERSION=1.31.0
ARG ORT_FLAVOR=cpu          # cpu | gpu
ARG TOKENIZERS_VERSION=1.27.0
ARG TARGETARCH=amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime is pinned: the Go binding needs API 31 and the int8 outputs
# change with the runtime version, so index and query must share this build.
# The GPU build also contains the CPU provider, and libonnxruntime.so does NOT
# link CUDA: providers are dlopen'd only when the EP is requested.
RUN set -eux; \
    if [ "$ORT_FLAVOR" = "gpu" ]; then \
      asset="onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}.tgz"; \
      dir="onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      asset="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"; \
      dir="onnxruntime-linux-aarch64-${ORT_VERSION}"; \
    else \
      asset="onnxruntime-linux-x64-${ORT_VERSION}.tgz"; \
      dir="onnxruntime-linux-x64-${ORT_VERSION}"; \
    fi; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${asset}" -o /tmp/ort.tgz; \
    tar -xzf /tmp/ort.tgz -C /opt; \
    mv "/opt/${dir}" /opt/onnxruntime

# HuggingFace tokenizers, statically linked (needed by cgo at build time).
RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then tok_arch="aarch64"; else tok_arch="amd64"; fi; \
    curl -fsSL "https://github.com/daulet/tokenizers/releases/download/v${TOKENIZERS_VERSION}/libtokenizers.linux-${tok_arch}.tar.gz" -o /tmp/lt.tgz; \
    mkdir -p /opt/libtokenizers; \
    tar -xzf /tmp/lt.tgz -C /opt/libtokenizers

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .

ENV CGO_ENABLED=1 CGO_LDFLAGS="-L/opt/libtokenizers"
RUN go build -trimpath -ldflags "-s -w" -o /out/argosctl ./cmd/argosctl \
    && go build -trimpath -ldflags "-s -w" -o /out/argosd ./cmd/argosd \
    && go build -trimpath -ldflags "-s -w" -o /out/argos-mcp ./cmd/argos-mcp

# --- runtime: CPU (default) -------------------------------------------------
FROM debian:13-slim AS runtime-cpu

ARG GIT_SHA=dev
LABEL org.opencontainers.image.revision=$GIT_SHA

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl poppler-utils tini libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=build /out/ /usr/local/bin/

ENV ARGOS_ORT_LIB=/usr/local/lib/libonnxruntime.so \
    HOME=/home/argos \
    ARGOS_HOME=/home/argos/.local/share/argos

RUN userdel -r ubuntu 2>/dev/null || true; \
    if ! id -u argos >/dev/null 2>&1; then useradd -m -u 1000 argos; fi
USER argos
WORKDIR /home/argos

# Models, base registry and config live here. They are NOT baked into the
# image: rebuilding the app must not re-download 1.2 GB of weights.
VOLUME ["/home/argos"]

EXPOSE 7712 7711
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["argosd", "--addr", "0.0.0.0:7712"]

# --- runtime: GPU (CUDA) ----------------------------------------------------
# Runs on CPU hosts too: libonnxruntime.so has no CUDA dependency and the
# provider is only loaded when the device policy asks for it. Bundling the
# CUDA runtime is what makes one image enough for CPU and NVIDIA.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime-gpu

ARG GIT_SHA=dev
LABEL org.opencontainers.image.revision=$GIT_SHA

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl poppler-utils tini libstdc++6 libgomp1 passwd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=build /opt/onnxruntime/lib/libonnxruntime_providers_*.so /usr/local/lib/
COPY --from=build /out/ /usr/local/bin/

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

# --- ONNX Runtime with OpenVINO (Intel CPU + iGPU) --------------------------
# Experimental: this compiles ONNX Runtime from source. Takes ~20-30 min and
# needs ~10 GB of build cache. The runtime image drops the toolchain.
FROM openvino/ubuntu24_dev:2026.4.1 AS ort-intel

ARG ORT_VERSION=1.31.0
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential python3 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "v${ORT_VERSION}" --recurse-submodules \
      --shallow-submodules https://github.com/microsoft/onnxruntime.git /ort

WORKDIR /ort
RUN ./build.sh --config Release --build_shared_lib --parallel --skip_tests \
      --skip_submodule_sync --use_openvino CPU \
      --allow_running_as_root --build_dir /build

FROM openvino/ubuntu24_runtime:2026.4.1 AS runtime-intel

ARG GIT_SHA=dev
LABEL org.opencontainers.image.revision=$GIT_SHA
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl poppler-utils tini libstdc++6 libgomp1 passwd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ort-intel /build/Release/libonnxruntime.so* /usr/local/lib/
COPY --from=ort-intel /build/Release/libonnxruntime_providers_*.so /usr/local/lib/
COPY --from=build /out/ /usr/local/bin/

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

# --- ONNX Runtime with MIGraphX (AMD) ---------------------------------------
# Experimental: ROCm ships no prebuilt ONNX Runtime, and the ROCm EP was
# removed from ORT; AMD recommends MIGraphX. Build takes ~45 min and the image
# is large (ROCm runtime included).
FROM rocm/dev-ubuntu-24.04:6.4 AS ort-amd

ARG ORT_VERSION=1.31.0
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential python3 migraphx-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "v${ORT_VERSION}" --recurse-submodules \
      --shallow-submodules https://github.com/microsoft/onnxruntime.git /ort

WORKDIR /ort
RUN ./build.sh --config Release --build_shared_lib --parallel --skip_tests \
      --skip_submodule_sync --use_migraphx --migraphx_home /opt/rocm \
      --allow_running_as_root --build_dir /build

FROM rocm/dev-ubuntu-24.04:6.4 AS runtime-amd

ARG GIT_SHA=dev
LABEL org.opencontainers.image.revision=$GIT_SHA
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl poppler-utils tini libstdc++6 libgomp1 passwd migraphx \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ort-amd /build/Release/libonnxruntime.so* /usr/local/lib/
COPY --from=ort-amd /build/Release/libonnxruntime_providers_*.so /usr/local/lib/
COPY --from=build /out/ /usr/local/bin/

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
