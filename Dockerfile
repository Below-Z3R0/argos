# syntax=docker/dockerfile:1.7

# --- build ------------------------------------------------------------------
FROM golang:1.27-bookworm AS build

ARG ORT_VERSION=1.31.0
ARG ORT_FLAVOR=cpu          # cpu | gpu
ARG TOKENIZERS_VERSION=1.27.0

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
    else \
      asset="onnxruntime-linux-x64-${ORT_VERSION}.tgz"; \
      dir="onnxruntime-linux-x64-${ORT_VERSION}"; \
    fi; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${asset}" -o /tmp/ort.tgz; \
    tar -xzf /tmp/ort.tgz -C /opt; \
    mv "/opt/${dir}" /opt/onnxruntime

# HuggingFace tokenizers, statically linked (needed by cgo at build time).
RUN curl -fsSL "https://github.com/daulet/tokenizers/releases/download/v${TOKENIZERS_VERSION}/libtokenizers.linux-amd64.tar.gz" -o /tmp/lt.tgz \
    && mkdir -p /opt/libtokenizers \
    && tar -xzf /tmp/lt.tgz -C /opt/libtokenizers

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
