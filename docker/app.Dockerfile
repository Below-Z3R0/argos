# syntax=docker/dockerfile:1.7
# Argos application layer. The runtime and ONNX Runtime come from BASE_IMAGE:
#   docker build -f docker/app.Dockerfile \
#     --build-arg BASE_IMAGE=argos-ort:cpu-1.31 \
#     -t argos:cpu .
#
# Updating Argos no longer recompiles ONNX Runtime.

ARG BASE_IMAGE=argos-ort:cpu-1.31

FROM golang:1.27-bookworm AS build

ARG TOKENIZERS_VERSION=1.27.0
ARG TARGETARCH=amd64
ARG GIT_SHA=dev

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

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

FROM ${BASE_IMAGE}

ARG GIT_SHA=dev
LABEL org.opencontainers.image.revision=$GIT_SHA
LABEL org.opencontainers.image.source="https://github.com/Below-Z3R0/argos"

COPY --from=build --chown=1000:1000 /out/ /usr/local/bin/
