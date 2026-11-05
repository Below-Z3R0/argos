BINARY := argosctl

.PHONY: build test lint fmt run clean

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
