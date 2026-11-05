// Command argos-mcp exposes the Argos CLI over the Model Context Protocol.
//
// It runs on the host: the tools shell out to argosctl, which owns validation,
// configuration and the Docker/Qdrant access. The server speaks Streamable HTTP
// so an agent connects by URL instead of spawning a stdio process.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/Below-Z3R0/argos/internal/mcp/daemon"
	"github.com/Below-Z3R0/argos/internal/mcp/jobs"
	"github.com/Below-Z3R0/argos/internal/mcp/runner"
	"github.com/Below-Z3R0/argos/internal/mcp/tools"
	"github.com/Below-Z3R0/argos/internal/metrics"
)

var version = "dev"

func main() {
	addr := flag.String("addr", "127.0.0.1:7711", "dirección de escucha")
	argosctl := flag.String("argosctl", "", "ruta al binario argosctl (default: PATH o ../argos/argosctl)")
	timeout := flag.Duration("timeout", 60*time.Second, "timeout por invocación del CLI")
	readOnly := flag.Bool("read-only", false, "deshabilita tools que escriben")
	argosd := flag.String("argosd", "http://127.0.0.1:7712", "URL de argosd para búsquedas (vacío usa el CLI)")
	showVersion := flag.Bool("version", false, "muestra la versión")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		return
	}

	bin, err := resolveBinary(*argosctl)
	if err != nil {
		log.Fatalf("argos-mcp: %v", err)
	}
	r := runner.New(bin, *timeout, 256<<10)
	jm := jobs.New(r, 20)

	srv := mcp.NewServer(&mcp.Implementation{Name: "argos-mcp", Version: version}, nil)
	reg := metrics.New()
	tools.Register(srv, tools.Deps{
		Runner:   r,
		Jobs:     jm,
		Daemon:   daemon.New(*argosd),
		Metrics:  reg,
		ReadOnly: *readOnly,
	})

	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return srv }, &mcp.StreamableHTTPOptions{JSONResponse: true})
	mux := http.NewServeMux()
	mux.Handle("/mcp", handler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintln(w, "ok")
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprint(w, reg.Render())
	})

	httpServer := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Printf("argos-mcp %s escuchando en http://%s/mcp (cli: %s, read-only: %v)", version, *addr, bin, *readOnly)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("argos-mcp: %v", err)
		}
	}()

	<-ctx.Done()
	shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdown)
}

// resolveBinary finds argosctl: explicit flag, PATH, then the sibling repo.
func resolveBinary(explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	if path, err := exec.LookPath("argosctl"); err == nil {
		return path, nil
	}
	home, err := os.UserHomeDir()
	if err == nil {
		candidate := filepath.Join(home, "Documents", "Github", "argos", "argosctl")
		if st, err := os.Stat(candidate); err == nil && !st.IsDir() {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("no encontré argosctl; pasá --argosctl /ruta/al/binario")
}
