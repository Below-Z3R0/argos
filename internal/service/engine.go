// Package service implements the long-running Argos engine.
//
// This is the piece that was missing: the CLI loads the model on every command,
// which is fine for administration and terrible for search. The engine keeps
// one embedder and one reranker per (model, dims) alive, so a request only pays
// the inference cost.
package service

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Below-Z3R0/argos/internal/config"
	dbpkg "github.com/Below-Z3R0/argos/internal/db"
	"github.com/Below-Z3R0/argos/internal/metrics"
	"github.com/Below-Z3R0/argos/internal/registry"
	"github.com/Below-Z3R0/argos/internal/types"
)

// Embedder is the subset of the embedding model the engine uses.
type Embedder interface {
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
	Dim() int
	Provider() string
}

// Store searches a collection.
type Store interface {
	Search(ctx context.Context, collection string, vector []float32, limit int, filter map[string]string) ([]types.Hit, error)
	Health(ctx context.Context) error
}

// Reranker scores query/passage pairs.
type Reranker interface {
	Score(ctx context.Context, query string, passages []string) ([]float32, error)
	Provider() string
}

// Factories build the heavy objects. Tests inject fakes; main injects ONNX.
type Factories struct {
	Embedder func(model registry.Model, dims int) (Embedder, error)
	Reranker func(model registry.Model) (Reranker, error)
}

// SearchRequest is the HTTP contract.
type SearchRequest struct {
	Query       string            `json:"query"`
	DB          string            `json:"db,omitempty"`
	K           int               `json:"k,omitempty"`
	MaxPerFile  int               `json:"max_per_file,omitempty"`
	MinScore    float64           `json:"min_score,omitempty"`
	Rerank      string            `json:"rerank,omitempty"` // auto | on | off
	RerankDepth int               `json:"rerank_depth,omitempty"`
	Filter      map[string]string `json:"filter,omitempty"`
}

// Hit is one result returned to the client.
type Hit struct {
	Path       string  `json:"path"`
	Heading    string  `json:"heading"`
	Text       string  `json:"text"`
	Score      float64 `json:"score"`
	Tokens     int     `json:"tokens"`
	ChunkIndex int     `json:"chunk_index"`
}

// SearchResponse is the HTTP response.
type SearchResponse struct {
	Query      string `json:"query"`
	DB         string `json:"db"`
	Collection string `json:"collection"`
	Model      string `json:"model"`
	ElapsedMS  int64  `json:"elapsed_ms"`
	Hits       []Hit  `json:"hits"`
}

// Target is the resolved base for a request.
type Target struct {
	Name       string
	Collection string
	ModelID    string
	Dims       int
}

// Engine holds the loaded models.
type Engine struct {
	cfg     config.Config
	catalog *registry.Catalog
	store   Store
	fact    Factories
	dbs     *dbpkg.Manager

	mu        sync.Mutex
	embedders map[string]Embedder
	rerankers map[string]Reranker
	states    map[string]*ModelStatus
	metrics   *metrics.Registry
	// use keeps a search from running while a model is being unloaded.
	use sync.RWMutex
}

// ModelStatus is the lifecycle of one model inside the service.
type ModelStatus struct {
	Key      string `json:"key"`
	State    string `json:"state"` // not_loaded | loading | ready | error
	Provider string `json:"provider,omitempty"`
	Error    string `json:"error,omitempty"`
	LoadMS   int64  `json:"load_ms,omitempty"`
}

// New creates an engine.
func New(cfg config.Config, catalog *registry.Catalog, store Store, fact Factories) *Engine {
	return &Engine{
		cfg:       cfg,
		catalog:   catalog,
		store:     store,
		fact:      fact,
		dbs:       dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs")),
		embedders: map[string]Embedder{},
		rerankers: map[string]Reranker{},
		states:    map[string]*ModelStatus{},
		metrics:   metrics.New(),
	}
}

// Resolve maps a base name (or "" for the config default) to its target.
func (e *Engine) Resolve(name string) (Target, error) {
	if name == "" {
		modelID := e.cfg.Defaults.Embedding
		m, err := e.catalog.Get(modelID)
		if err != nil {
			return Target{}, err
		}
		dims := e.cfg.Index.Dims
		if dims == 0 {
			dims = m.Specs.Dims
		}
		return Target{Name: "default", Collection: e.cfg.Index.Collection, ModelID: modelID, Dims: dims}, nil
	}
	meta, err := e.dbs.Get(name)
	if err != nil {
		return Target{}, err
	}
	return Target{Name: meta.Name, Collection: meta.Collection, ModelID: meta.Model, Dims: meta.Dims}, nil
}

// Search runs a query against a base.
func (e *Engine) Search(ctx context.Context, req SearchRequest) (SearchResponse, error) {
	e.use.RLock()
	defer e.use.RUnlock()
	start := time.Now()
	defer func() {
		e.metrics.Observe("argos_search_seconds", time.Since(start).Seconds(), nil)
	}()
	if strings.TrimSpace(req.Query) == "" {
		return SearchResponse{}, fmt.Errorf("query vacía")
	}
	if req.K <= 0 {
		req.K = 20
	}
	if req.MaxPerFile <= 0 {
		req.MaxPerFile = 2
	}
	if req.Rerank == "" {
		req.Rerank = "auto"
	}

	target, err := e.Resolve(req.DB)
	if err != nil {
		return SearchResponse{}, err
	}
	emb, err := e.embedderFor(target)
	if err != nil {
		return SearchResponse{}, err
	}
	vec, err := emb.EmbedQuery(ctx, req.Query)
	if err != nil {
		return SearchResponse{}, err
	}

	pool := req.K * 3
	hits, err := e.store.Search(ctx, target.Collection, vec, pool, req.Filter)
	if err != nil {
		return SearchResponse{}, err
	}
	results := toHits(hits)

	if req.Rerank != "off" {
		rr, rerr := e.rerankerFor()
		if rerr == nil && rr != nil {
			rerankStart := time.Now()
			defer func() {
				e.metrics.Observe("argos_rerank_seconds", time.Since(rerankStart).Seconds(), nil)
			}()
			depth := req.RerankDepth
			if depth <= 0 || depth > len(results) {
				depth = len(results)
			}
			if depth > 0 {
				passages := make([]string, depth)
				for i := 0; i < depth; i++ {
					passages[i] = results[i].Text
				}
				scores, serr := rr.Score(ctx, req.Query, passages)
				if serr == nil && len(scores) == depth {
					for i := 0; i < depth; i++ {
						results[i].Score = float64(scores[i])
					}
					sort.SliceStable(results[:depth], func(i, j int) bool {
						return results[i].Score > results[j].Score
					})
				}
			}
		} else if req.Rerank == "on" {
			return SearchResponse{}, fmt.Errorf("reranker no disponible: %w", rerr)
		}
	}

	out := make([]Hit, 0, req.K)
	perFile := map[string]int{}
	for _, r := range results {
		if req.MinScore > 0 && r.Score < req.MinScore {
			continue
		}
		if perFile[r.Path] >= req.MaxPerFile {
			continue
		}
		perFile[r.Path]++
		out = append(out, r)
		if len(out) == req.K {
			break
		}
	}

	e.metrics.Inc("argos_search_total", metrics.Labels{"db": target.Name, "model": target.ModelID})
	e.metrics.Add("argos_hits_total", float64(len(out)), nil)
	return SearchResponse{
		Query:      req.Query,
		DB:         target.Name,
		Collection: target.Collection,
		Model:      target.ModelID,
		ElapsedMS:  time.Since(start).Milliseconds(),
		Hits:       out,
	}, nil
}

// Status describes what is loaded.
type Status struct {
	Collection string        `json:"collection"`
	Model      string        `json:"model"`
	Embedders  []string      `json:"embedders_loaded"`
	Rerankers  []string      `json:"rerankers_loaded"`
	Provider   string        `json:"provider"`
	Models     []ModelStatus `json:"models"`
	Ready      bool          `json:"ready"`
}

// Status reports the live state.
func (e *Engine) Status(ctx context.Context) Status {
	e.mu.Lock()
	defer e.mu.Unlock()
	st := Status{Collection: e.cfg.Index.Collection, Model: e.cfg.Defaults.Embedding}
	for k, v := range e.embedders {
		st.Embedders = append(st.Embedders, k)
		st.Provider = v.Provider()
	}
	for k := range e.rerankers {
		st.Rerankers = append(st.Rerankers, k)
	}
	sort.Strings(st.Embedders)
	sort.Strings(st.Rerankers)
	for _, state := range e.states {
		st.Models = append(st.Models, *state)
	}
	sort.Slice(st.Models, func(i, j int) bool { return st.Models[i].Key < st.Models[j].Key })
	st.Ready = e.store.Health(ctx) == nil
	return st
}

// Load loads one model ahead of time: embed, rerank or all.
func (e *Engine) Load(ctx context.Context, kind string) error {
	e.use.Lock()
	defer e.use.Unlock()
	switch kind {
	case "embed":
		_, err := e.embedderFor(Target{ModelID: e.cfg.Defaults.Embedding, Dims: e.defaultDims()})
		return err
	case "rerank":
		_, err := e.rerankerFor()
		return err
	case "all":
		if _, err := e.embedderFor(Target{ModelID: e.cfg.Defaults.Embedding, Dims: e.defaultDims()}); err != nil {
			return err
		}
		_, err := e.rerankerFor()
		return err
	default:
		return fmt.Errorf("modelo desconocido %q (embed|rerank|all)", kind)
	}
}

// Unload closes the sessions and frees their memory. A running search finishes
// first: the RWMutex serializes the two.
func (e *Engine) Unload(kind string) {
	e.use.Lock()
	defer e.use.Unlock()

	dropEmbed := kind == "embed" || kind == "all"
	dropRerank := kind == "rerank" || kind == "all"

	e.mu.Lock()
	defer e.mu.Unlock()
	if dropEmbed {
		for key, emb := range e.embedders {
			if c, ok := emb.(interface{ Close() error }); ok {
				_ = c.Close()
			}
			delete(e.embedders, key)
			if st, ok := e.states[key]; ok {
				st.State = "not_loaded"
				st.Provider = ""
				st.LoadMS = 0
			}
		}
	}
	if dropRerank {
		for id, rr := range e.rerankers {
			if c, ok := rr.(interface{ Close() error }); ok {
				_ = c.Close()
			}
			delete(e.rerankers, id)
			key := "rerank:" + id
			if st, ok := e.states[key]; ok {
				st.State = "not_loaded"
				st.Provider = ""
				st.LoadMS = 0
			}
		}
	}
	e.metrics.Set("argos_models_loaded", float64(len(e.embedders)), metrics.Labels{"kind": "embed"})
	e.metrics.Set("argos_models_loaded", float64(len(e.rerankers)), metrics.Labels{"kind": "rerank"})
	e.metrics.Inc("argos_model_unload_total", metrics.Labels{"model": kind})
	trimHeap()
}

// Metrics renders the Prometheus exposition.
func (e *Engine) Metrics() string { return e.metrics.Render() }

// Ready reports whether Qdrant answers.
func (e *Engine) Ready(ctx context.Context) error { return e.store.Health(ctx) }

// Close releases the loaded models.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, v := range e.embedders {
		if c, ok := v.(interface{ Close() error }); ok {
			_ = c.Close()
		}
	}
	for _, v := range e.rerankers {
		if c, ok := v.(interface{ Close() error }); ok {
			_ = c.Close()
		}
	}
	e.embedders = map[string]Embedder{}
	e.rerankers = map[string]Reranker{}
}

func (e *Engine) defaultDims() int {
	if e.cfg.Index.Dims > 0 {
		return e.cfg.Index.Dims
	}
	if m, err := e.catalog.Get(e.cfg.Defaults.Embedding); err == nil {
		return m.Specs.Dims
	}
	return 0
}

func (e *Engine) embedderFor(t Target) (Embedder, error) {
	key := fmt.Sprintf("embed:%s/%d", t.ModelID, t.Dims)
	e.mu.Lock()
	defer e.mu.Unlock()
	if emb, ok := e.embedders[key]; ok {
		return emb, nil
	}
	state := &ModelStatus{Key: key, State: "loading"}
	e.states[key] = state
	start := time.Now()

	m, err := e.catalog.Get(t.ModelID)
	if err == nil {
		var emb Embedder
		emb, err = e.fact.Embedder(m, t.Dims)
		if err == nil {
			e.embedders[key] = emb
			state.State = "ready"
			state.Provider = emb.Provider()
			state.LoadMS = time.Since(start).Milliseconds()
			e.metrics.Observe("argos_model_load_seconds", time.Since(start).Seconds(), metrics.Labels{"model": t.ModelID})
			e.metrics.Inc("argos_model_load_total", metrics.Labels{"model": t.ModelID, "state": "ready"})
			e.metrics.Set("argos_models_loaded", float64(len(e.embedders)), metrics.Labels{"kind": "embed"})
			return emb, nil
		}
	}
	state.State = "error"
	state.Error = err.Error()
	e.metrics.Inc("argos_model_load_total", metrics.Labels{"model": t.ModelID, "state": "error"})
	return nil, err
}

func (e *Engine) rerankerFor() (Reranker, error) {
	id := e.cfg.Defaults.Reranker
	key := "rerank:" + id
	e.mu.Lock()
	defer e.mu.Unlock()
	if rr, ok := e.rerankers[id]; ok {
		return rr, nil
	}
	state := &ModelStatus{Key: key, State: "loading"}
	e.states[key] = state
	start := time.Now()

	m, err := e.catalog.Get(id)
	if err == nil {
		var rr Reranker
		rr, err = e.fact.Reranker(m)
		if err == nil {
			e.rerankers[id] = rr
			state.State = "ready"
			state.Provider = rr.Provider()
			state.LoadMS = time.Since(start).Milliseconds()
			e.metrics.Observe("argos_model_load_seconds", time.Since(start).Seconds(), metrics.Labels{"model": id})
			e.metrics.Inc("argos_model_load_total", metrics.Labels{"model": id, "state": "ready"})
			e.metrics.Set("argos_models_loaded", float64(len(e.rerankers)), metrics.Labels{"kind": "rerank"})
			return rr, nil
		}
	}
	state.State = "error"
	state.Error = err.Error()
	e.metrics.Inc("argos_model_load_total", metrics.Labels{"model": id, "state": "error"})
	return nil, err
}

func toHits(in []types.Hit) []Hit {
	out := make([]Hit, 0, len(in))
	for _, h := range in {
		out = append(out, Hit{
			Path:       str(h.Payload["path"]),
			Heading:    str(h.Payload["heading"]),
			Text:       str(h.Payload["text"]),
			Score:      float64(h.Score),
			Tokens:     num(h.Payload["tokens"]),
			ChunkIndex: num(h.Payload["chunk_index"]),
		})
	}
	return out
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func num(v any) int {
	switch n := v.(type) {
	case int64:
		return int(n)
	case int:
		return n
	case float64:
		return int(n)
	default:
		return 0
	}
}
