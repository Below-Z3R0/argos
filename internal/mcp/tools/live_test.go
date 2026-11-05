package tools

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestLiveServer talks to a running argos-mcp. Set ARGOS_MCP_URL to enable it:
//
//	ARGOS_MCP_URL=http://127.0.0.1:7711 go test ./internal/tools -run TestLiveServer
func TestLiveServer(t *testing.T) {
	endpoint := os.Getenv("ARGOS_MCP_URL")
	if endpoint == "" {
		t.Skip("set ARGOS_MCP_URL to run against a live server")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := mcp.NewClient(&mcp.Implementation{Name: "live", Version: "0"}, nil)
	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: endpoint}, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer session.Close()

	list, err := session.ListTools(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(list.Tools) < 10 {
		t.Fatalf("tools: %d", len(list.Tools))
	}

	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "search",
		Arguments: map[string]any{"query": "api endpoints", "db": "knowledge", "k": 3, "rerank": "off"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.IsError {
		t.Fatalf("search error: %s", contentText(res))
	}
	out := contentText(res)
	if !strings.Contains(out, "hits") {
		t.Fatalf("unexpected: %s", out)
	}
	if len(out) > 400 {
		out = out[:400]
	}
	t.Logf("search devolvió: %s", out)

	res, err = session.CallTool(ctx, &mcp.CallToolParams{Name: "doctor"})
	if err != nil {
		t.Fatal(err)
	}
	if res.IsError {
		t.Fatalf("doctor error: %s", contentText(res))
	}
	if !strings.Contains(contentText(res), "config") {
		t.Fatalf("doctor: %s", contentText(res))
	}
}
