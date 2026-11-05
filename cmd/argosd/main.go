// Command argosd is the long-running Argos service.
//
// It keeps the embedding and reranking models loaded and serves search over
// HTTP. The CLI stays for administration; this is what makes repeated search
// calls cost milliseconds instead of reloading a model each time.
package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/Below-Z3R0/argos/internal/api"
	"github.com/Below-Z3R0/argos/internal/config"
	"github.com/Below-Z3R0/argos/internal/embed"
	"github.com/Below-Z3R0/argos/internal/models"
	"github.com/Below-Z3R0/argos/internal/registry"
	"github.com/Below-Z3R0/argos/internal/rerank"
	"github.com/Below-Z3R0/argos/internal/service"
	qstore "github.com/Below-Z3R0/argos/internal/store/qdrant"
	"github.com/Below-Z3R0/argos/internal/version"
)

func main() {
	addr := flagValue("addr", "127.0.0.1:7712", "dirección de escucha")
	preload := flagValue("preload", "none", "carga modelos al arrancar: none|embed|rerank|all")
	if addr == "" {
		return
	}
	cfgPath, err := config.FilePath()
	if err != nil {
		log.Fatalf("argosd: %v", err)
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("argosd: %v", err)
	}
	catalog, err := registry.Load()
	if err != nil {
		log.Fatalf("argosd: %v", err)
	}

	host, portStr, err := net.SplitHostPort(cfg.Qdrant.URL)
	if err != nil {
		log.Fatalf("argosd: qdrant.url: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		log.Fatalf("argosd: qdrant.url: %v", err)
	}
	store, err := qstore.NewWithREST(host, port, cfg.Qdrant.RESTURL, cfg.Qdrant.SnapshotsPath)
	if err != nil {
		log.Fatalf("argosd: %v", err)
	}
	defer store.Close()

	modelStore := models.New(filepath.Join(cfg.Paths.Home, "models"))
	lib := embed.DiscoverRuntime(cfg.Paths.Home)

	factories := service.Factories{
		Embedder: func(m registry.Model, dims int) (service.Embedder, error) {
			dir, err := modelStore.Dir(m.Task, m.ID)
			if err != nil {
				return nil, err
			}
			if _, ok := modelStore.Status(m.Task, m.ID); !ok {
				return nil, fmt.Errorf("%s no está instalado", m.ID)
			}
			e, err := embed.NewORT(embed.ORTOptions{
				ModelPath:     filepath.Join(dir, "model.onnx"),
				TokenizerPath: filepath.Join(dir, "tokenizer.json"),
				LibraryPath:   lib,
				Name:          m.ID,
				Dims:          dims,
				MaxTokens:     m.Specs.MaxTokens,
				Pooling:       m.Specs.Pooling,
				QueryInstruct: m.QueryInstruction,
				Prefixes:      m.Prefixes,
				Threads:       cfg.Resources.Threads,
				BatchSize:     cfg.Resources.Batch,
				Device:        cfg.Resources.Devices.Embed,
				GPUIndex:      -1,
			})
			if err != nil {
				return nil, err
			}
			return e, nil
		},
		Reranker: func(m registry.Model) (service.Reranker, error) {
			dir, err := modelStore.Dir(m.Task, m.ID)
			if err != nil {
				return nil, err
			}
			if _, ok := modelStore.Status(m.Task, m.ID); !ok {
				return nil, fmt.Errorf("%s no está instalado", m.ID)
			}
			return rerank.New(rerank.Options{
				ModelPath:     filepath.Join(dir, "model.onnx"),
				TokenizerPath: filepath.Join(dir, "tokenizer.json"),
				LibraryPath:   lib,
				Device:        cfg.Resources.Devices.Rerank,
				GPUIndex:      -1,
				Threads:       cfg.Resources.Threads,
			})
		},
	}

	engine := service.New(cfg, catalog, store, factories)
	defer engine.Close()

	if preload != "" && preload != "none" {
		start := time.Now()
		if err := engine.Load(context.Background(), preload); err != nil {
			log.Printf("argosd: preload %s falló: %v", preload, err)
		} else {
			log.Printf("argosd: preload %s en %d ms", preload, time.Since(start).Milliseconds())
		}
	}

	server := &http.Server{
		Addr:              addr,
		Handler:           api.New(engine).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		log.Printf("argosd %s escuchando en http://%s (colección %s)", version.String(), addr, cfg.Index.Collection)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("argosd: %v", err)
		}
	}()
	<-ctx.Done()
	shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = server.Shutdown(shutdown)
}

// flagValue is a tiny wrapper so main stays readable.
func flagValue(name, def, usage string) string {
	value := def
	for i, a := range os.Args[1:] {
		if a == "--"+name && i+2 <= len(os.Args)-1 {
			value = os.Args[i+2]
		}
		if a == "--help" || a == "-h" {
			fmt.Printf("argosd — servicio de búsqueda de Argos\n\n  --%s string\n    %s (default %q)\n", name, usage, def)
			return ""
		}
	}
	return value
}
