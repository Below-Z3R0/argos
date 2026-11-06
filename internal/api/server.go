// Package api exposes the engine over HTTP.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Below-Z3R0/argos/internal/service"
)

// Engine is the subset of the service the HTTP layer uses.
type Engine interface {
	Search(ctx context.Context, req service.SearchRequest) (service.SearchResponse, error)
	Status(ctx context.Context) service.Status
	Metrics() string
	Ready(ctx context.Context) error
	Load(ctx context.Context, kind string) error
	LoadModel(ctx context.Context, id string) error
	Unload(kind string)
	UnloadModel(id string) error
}

// Server wires the handlers.
type Server struct {
	engine Engine
}

// New returns a server.
func New(e Engine) *Server { return &Server{engine: e} }

// Handler returns the routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/search", s.search)
	mux.HandleFunc("/v1/status", s.status)
	mux.HandleFunc("/v1/models/load", s.load)
	mux.HandleFunc("/v1/models/unload", s.unload)
	mux.HandleFunc("/metrics", s.metrics)
	mux.HandleFunc("/readyz", s.readyz)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintln(w, "ok")
	})
	return mux
}

func (s *Server) search(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "método no permitido", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()
	var req service.SearchRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "JSON inválido: "+err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		http.Error(w, "query vacía", http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()
	resp, err := s.engine.Search(ctx, req)
	if err != nil {
		status := http.StatusInternalServerError
		switch {
		case errors.Is(err, service.ErrInvalidRequest):
			status = http.StatusBadRequest
		case errors.Is(err, service.ErrModelUnavailable):
			status = http.StatusServiceUnavailable
		case errors.Is(err, service.ErrStoreUnavailable):
			status = http.StatusBadGateway
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			status = http.StatusGatewayTimeout
		}
		http.Error(w, err.Error(), status)
		return
	}
	writeJSON(w, resp)
}

func (s *Server) status(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.engine.Status(r.Context()))
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) metrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprint(w, s.engine.Metrics())
}

func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := s.engine.Ready(ctx); err != nil {
		http.Error(w, "no listo: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	fmt.Fprintln(w, "ready")
}

type modelAction struct {
	Model string `json:"model"`
}

func (s *Server) load(w http.ResponseWriter, r *http.Request) {
	s.modelAction(w, r, true)
}

func (s *Server) unload(w http.ResponseWriter, r *http.Request) {
	s.modelAction(w, r, false)
}

func (s *Server) modelAction(w http.ResponseWriter, r *http.Request, load bool) {
	if r.Method != http.MethodPost {
		http.Error(w, "método no permitido", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()
	var req modelAction
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&req); err != nil {
		http.Error(w, "JSON inválido: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Model == "" {
		req.Model = "all"
	}
	if load {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
		defer cancel()
		var err error
		switch req.Model {
		case "embed", "rerank", "all":
			err = s.engine.Load(ctx, req.Model)
		default:
			err = s.engine.LoadModel(ctx, req.Model)
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
	} else {
		switch req.Model {
		case "embed", "rerank", "all":
			s.engine.Unload(req.Model)
		default:
			if err := s.engine.UnloadModel(req.Model); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
	}
	writeJSON(w, map[string]any{"model": req.Model, "loaded": load})
}
