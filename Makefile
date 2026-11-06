BINARY := argosctl

.PHONY: build test lint fmt run clean image image-cpu image-cuda image-intel image-amd

# Build metadata, same variables the container image embeds.
SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo none)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
LDFLAGS := -s -w \
	-X github.com/Below-Z3R0/argos/internal/version.Version=$(SHA) \
	-X github.com/Below-Z3R0/argos/internal/version.Commit=$(COMMIT) \
	-X github.com/Below-Z3R0/argos/internal/version.Date=$(BUILD_DATE)

build:
	go build -trimpath -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/argosctl

test:
	go test ./...

lint:
	go vet ./... && test -z "$$(gofmt -l .)"

fmt:
	gofmt -w .

run: build
	./$(BINARY) $(ARGS)

clean:
	rm -f $(BINARY)

# App image only. ONNX Runtime and the provider live in a published base:
#   ghcr.io/below-z3r0/b3l0w-ort:<variant>-1.31
VARIANT ?= cpu
ORT_VERSION ?= 1.31
BASE_IMAGE ?= ghcr.io/below-z3r0/b3l0w-ort:$(VARIANT)-$(ORT_VERSION)
IMAGE ?= argos:local-$(VARIANT)-$(SHA)

image:
	docker build -f docker/app.Dockerfile \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg GIT_SHA=$(COMMIT) \
		--build-arg VERSION=$(SHA) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		-t $(IMAGE) .

image-cpu:
	$(MAKE) image VARIANT=cpu IMAGE=argos:cpu

image-cuda:
	$(MAKE) image VARIANT=cuda IMAGE=argos:cuda

image-intel:
	$(MAKE) image VARIANT=intel IMAGE=argos:intel

image-amd:
	$(MAKE) image VARIANT=amd IMAGE=argos:amd
