BINARY := argosctl

.PHONY: build test lint fmt run clean image image-cpu image-gpu

build:
	go build -trimpath -ldflags "-s -w" -o $(BINARY) ./cmd/argosctl

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

IMAGE ?= argos:local
TARGET ?= runtime-gpu
ORT_FLAVOR ?= gpu

# Build local de la imagen unificada (CPU + NVIDIA).
image:
	docker build --target $(TARGET) --build-arg ORT_FLAVOR=$(ORT_FLAVOR) -t $(IMAGE) .

image-cpu:
	$(MAKE) image TARGET=runtime-cpu ORT_FLAVOR=cpu IMAGE=argos:cpu

image-gpu:
	$(MAKE) image TARGET=runtime-gpu ORT_FLAVOR=gpu IMAGE=argos:gpu
