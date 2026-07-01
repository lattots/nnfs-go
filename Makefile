.PHONY: run test test-v

run:
	@go run ./cmd/main.go

test:
	@go test ./pkg/neural_net

test-v:
	@go test -v ./pkg/neural_net
