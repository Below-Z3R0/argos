package tools

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/Below-Z3R0/argos/internal/mcp/daemon"
	"github.com/Below-Z3R0/argos/internal/mcp/jobs"
	"github.com/Below-Z3R0/argos/internal/mcp/runner"
)

func fakeArgosctl(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	bin := filepath.Join(dir, "argosctl")
	script := `#!/bin/sh
case "$1" in
  search) echo '{"hits":[{"path":"a.md","score":0.9}]}' ;;
  doctor) echo "todo en orden" ;;
  config) if [ "$2" = "get" ]; then echo "valor"; else echo "configurado $3"; fi ;;
  *) echo "argosctl: $@" ;;
esac
`
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin
}

func TestToolsOverMCP(t *testing.T) {
	r := runner.New(fakeArgosctl(t), 5*time.Second, 1<<20)
	srv := mcp.NewServer(&mcp.Implementation{Name: "test", Version: "0"}, nil)
	Register(srv, Deps{Runner: r, Jobs: jobs.New(r, 5)})

	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return srv }, nil)
	ts := httptest.NewServer(handler)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	client := mcp.NewClient(&mcp.Implementation{Name: "client", Version: "0"}, nil)
	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: ts.URL}, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer session.Close()

	tools, err := session.ListTools(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools.Tools) < 10 {
		t.Fatalf("expected the full tool set, got %d", len(tools.Tools))
	}

	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "search",
		Arguments: map[string]any{"query": "hola", "db": "knowledge", "k": 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	text := contentText(res)
	if !strings.Contains(text, "a.md") {
		t.Fatalf("search result: %q", text)
	}

	res, err = session.CallTool(ctx, &mcp.CallToolParams{Name: "doctor"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(contentText(res), "todo en orden") {
		t.Fatalf("doctor result: %q", contentText(res))
	}
}

func TestReadOnlyBlocksWrites(t *testing.T) {
	r := runner.New(fakeArgosctl(t), 5*time.Second, 1<<20)
	srv := mcp.NewServer(&mcp.Implementation{Name: "test", Version: "0"}, nil)
	Register(srv, Deps{Runner: r, Jobs: jobs.New(r, 5), ReadOnly: true})

	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return srv }, nil)
	ts := httptest.NewServer(handler)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	client := mcp.NewClient(&mcp.Implementation{Name: "client", Version: "0"}, nil)
	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: ts.URL}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "config_set",
		Arguments: map[string]any{"key": "index.dims", "value": "512"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !res.IsError || !strings.Contains(contentText(res), "solo lectura") {
		t.Fatalf("expected read-only error, got %+v", res)
	}
}

func TestConfigSetRejectsUnknownKey(t *testing.T) {
	if allowedKey("some.unknown") {
		t.Fatal("unknown key allowed")
	}
	if !allowedKey("index.dims") {
		t.Fatal("known key rejected")
	}
}

func contentText(res *mcp.CallToolResult) string {
	var b strings.Builder
	for _, c := range res.Content {
		if t, ok := c.(*mcp.TextContent); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

func TestSearchUsesDaemonWhenAvailable(t *testing.T) {
	daemonSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/search" {
			t.Errorf("path %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"collection":"c","hits":[{"path":"desde-argosd.md"}]}`))
	}))
	defer daemonSrv.Close()

	r := runner.New(fakeArgosctl(t), 5*time.Second, 1<<20)
	srv := mcp.NewServer(&mcp.Implementation{Name: "test", Version: "0"}, nil)
	Register(srv, Deps{Runner: r, Jobs: jobs.New(r, 5), Daemon: daemon.New(daemonSrv.URL)})

	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return srv }, nil)
	ts := httptest.NewServer(handler)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	client := mcp.NewClient(&mcp.Implementation{Name: "client", Version: "0"}, nil)
	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: ts.URL}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "search", Arguments: map[string]any{"query": "x"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(contentText(res), "desde-argosd.md") {
		t.Fatalf("daemon not used: %s", contentText(res))
	}
}
