// Package service implements the long-running Argos engine.
//
// This is the piece that was missing: the CLI loads the model on every command,
// which is fine for administration and terrible for search. The engine keeps
// one embedder and one reranker per (model, dims) alive, so a request only pays
// the inference cost.
package service

import (
	"context"
	"errors"
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
	"github.com/Below-Z3R0/argos/internal/resources"
	"github.com/Below-Z3R0/argos/internal/sparse"
	"github.com/Below-Z3R0/argos/internal/types"
)

// Errors the HTTP layer maps to status codes. Everything else is a 500.
var (
	ErrInvalidRequest   = errors.New("invalid request")
	ErrModelUnavailable = errors.New("model unavailable")
	ErrStoreUnavailable = errors.New("store unavailable")
)

const ramHeadroomMB = 256

// Embedder is the subset of the embedding model the engine uses.
type Embedder interface {
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
	Dim() int
	Provider() string
}

// Store searches a collection.
type Store interface {
	Search(ctx context.Context, collection string, vector []float32, limit int, filter map[string]string) ([]types.Hit, error)
	SearchHybrid(ctx context.Context, collection string, dense []float32, sparseIdx []uint32, sparseVal []float32, limit int, filter map[string]string) ([]types.Hit, error)
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
	DBs         []string          `json:"dbs,omitempty"`
	Model       string            `json:"model,omitempty"`
	ExcludeDBs  []string          `json:"exclude_dbs,omitempty"`
	K           int               `json:"k,omitempty"`
	MaxPerFile  int               `json:"max_per_file,omitempty"`
	MinScore    float64           `json:"min_score,omitempty"`
	Rerank      string            `json:"rerank,omitempty"` // auto | on | off
	Mode        string            `json:"mode,omitempty"`   // dense | hybrid (empty = config)
	RerankDepth int               `json:"rerank_depth,omitempty"`
	Filter      map[string]string `json:"filter,omitempty"`
}

// Hit is one result returned to the client.
type Hit struct {
	DB         string  `json:"db,omitempty"`
	Path       string  `json:"path"`
	Heading    string  `json:"heading"`
	Text       string  `json:"text"`
	Score      float64 `json:"score"`
	Tokens     int     `json:"tokens"`
	ChunkIndex int     `json:"chunk_index"`
}

// SearchResponse is the HTTP response.
type SearchResponse struct {
	Query      string   `json:"query"`
	DB         string   `json:"db"`
	Collection string   `json:"collection"`
	Model      string   `json:"model"`
	Mode       string   `json:"mode"`
	ElapsedMS  int64    `json:"elapsed_ms"`
	Hits       []Hit    `json:"hits"`
	Warnings   []string `json:"warnings,omitempty"`
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
	// loadLocks deduplicates concurrent loads of the same model without
	// holding e.mu during the expensive ONNX session creation.
	loadLocks map[string]*sync.Mutex
	// ramAvailable is injectable for tests; nil means read the host.
	ramAvailable func() int
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
		cfg:          cfg,
		catalog:      catalog,
		store:        store,
		fact:         fact,
		dbs:          dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs")),
		embedders:    map[string]Embedder{},
		rerankers:    map[string]Reranker{},
		states:       map[string]*ModelStatus{},
		loadLocks:    map[string]*sync.Mutex{},
		metrics:      metrics.New(),
		ramAvailable: resources.AvailableRAMMB,
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
			dims = m.EffectiveDims()
		}
		return Target{Name: "default", Collection: e.cfg.Index.Collection, ModelID: modelID, Dims: dims}, nil
	}
	meta, err := e.dbs.Get(name)
	if err != nil {
		return Target{}, err
	}
	return Target{Name: meta.Name, Collection: meta.Collection, ModelID: meta.Model, Dims: meta.Dims}, nil
}

// targets resolves the selector: one base, a comma list, all bases, and the
// optional model/exclude filters. A plain DB keeps the historical behavior.
func (e *Engine) targets(req SearchRequest) ([]Target, error) {
	selectAll := req.DB == "all"
	names := make([]string, 0)
	if req.DB != "" && req.DB != "all" {
		for _, n := range strings.Split(req.DB, ",") {
			if n = strings.TrimSpace(n); n != "" {
				names = append(names, n)
			}
		}
	}
	names = append(names, req.DBs...)
	if !selectAll && len(names) == 0 && req.Model == "" && len(req.ExcludeDBs) == 0 {
		target, err := e.Resolve("")
		if err != nil {
			return nil, err
		}
		return []Target{target}, nil
	}
	if len(names) == 1 && !selectAll && req.Model == "" && len(req.ExcludeDBs) == 0 {
		target, err := e.Resolve(names[0])
		if err != nil {
			return nil, err
		}
		return []Target{target}, nil
	}
	list, err := e.dbs.List()
	if err != nil {
		return nil, err
	}
	if len(list) == 0 {
		return nil, fmt.Errorf("no hay bases registradas")
	}
	wanted := map[string]bool{}
	for _, n := range names {
		wanted[n] = true
	}
	excluded := map[string]bool{}
	for _, n := range req.ExcludeDBs {
		excluded[strings.TrimSpace(n)] = true
	}
	targets := make([]Target, 0, len(list))
	for _, meta := range list {
		if len(wanted) > 0 && !wanted[meta.Name] {
			continue
		}
		if req.Model != "" && req.Model != "all" && meta.Model != req.Model {
			continue
		}
		if excluded[meta.Name] {
			continue
		}
		targets = append(targets, Target{Name: meta.Name, Collection: meta.Collection, ModelID: meta.Model, Dims: meta.Dims})
	}
	if len(targets) == 0 {
		return nil, fmt.Errorf("%w: ningún target coincide con el selector", ErrInvalidRequest)
	}
	return targets, nil
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
	switch req.Mode {
	case "":
		if !e.cfg.Search.Hybrid {
			req.Mode = "dense"
		} else {
			req.Mode = "hybrid"
		}
	case "dense", "hybrid":
	default:
		return SearchResponse{}, fmt.Errorf("%w: search.mode %q no es dense|hybrid", ErrInvalidRequest, req.Mode)
	}

	pool := req.K * 3
	targets, err := e.targets(req)
	if err != nil {
		return SearchResponse{}, err
	}

	sv := sparse.Encode(req.Query)
	ranked := make([][]types.Hit, 0, len(targets))
	queryVectors := map[string][]float32{}
	for _, target := range targets {
		emb, err := e.embedderFor(target)
		if err != nil {
			return SearchResponse{}, err
		}
		vecKey := fmt.Sprintf("%s/%d", target.ModelID, target.Dims)
		vec, ok := queryVectors[vecKey]
		if !ok {
			vec, err = emb.EmbedQuery(ctx, req.Query)
			if err != nil {
				return SearchResponse{}, fmt.Errorf("%w: %v", ErrModelUnavailable, err)
			}
			queryVectors[vecKey] = vec
		}
		var hits []types.Hit
		if req.Mode == "dense" {
			hits, err = e.store.Search(ctx, target.Collection, vec, pool, req.Filter)
		} else {
			hits, err = e.store.SearchHybrid(ctx, target.Collection, vec, sv.Indices, sv.Values, pool, req.Filter)
		}
		if err != nil {
			return SearchResponse{}, fmt.Errorf("%w: %v", ErrStoreUnavailable, err)
		}
		ranked = append(ranked, hits)
	}
	results := fuseHits(targets, ranked)
	var warnings []string

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
				} else if serr != nil {
					warnings = append(warnings, "reranker: "+serr.Error())
				}
			}
		} else if req.Rerank == "on" {
			return SearchResponse{}, fmt.Errorf("%w: reranker no disponible: %v", ErrModelUnavailable, rerr)
		} else if rerr != nil {
			warnings = append(warnings, "reranker no disponible: "+rerr.Error())
		}
	}

	out := make([]Hit, 0, req.K)
	perFile := map[string]int{}
	for _, r := range results {
		if req.MinScore > 0 && r.Score < req.MinScore {
			continue
		}
		fileKey := r.DB + "|" + r.Path
		if perFile[fileKey] >= req.MaxPerFile {
			continue
		}
		perFile[fileKey]++
		out = append(out, r)
		if len(out) == req.K {
			break
		}
	}

	dbLabel, modelLabel := "all", "mixed"
	if len(targets) == 1 {
		dbLabel, modelLabel = targets[0].Name, targets[0].ModelID
	}
	e.metrics.Inc("argos_search_total", metrics.Labels{"db": dbLabel, "model": modelLabel})
	e.metrics.Add("argos_hits_total", float64(len(out)), nil)
	resp := SearchResponse{Query: req.Query, Mode: req.Mode, ElapsedMS: time.Since(start).Milliseconds(), Hits: out, Warnings: warnings}
	if len(targets) == 1 {
		resp.DB, resp.Collection, resp.Model = targets[0].Name, targets[0].Collection, targets[0].ModelID
	} else {
		resp.DB, resp.Collection, resp.Model = "all", "all", "mixed"
	}
	return resp, nil
}

// fuseHits merges per-database rankings with reciprocal rank fusion. A chunk
// that appears in two bases adds both contributions, which is intentional:
// agreement between independent indexes is a stronger signal.
func fuseHits(targets []Target, ranked [][]types.Hit) []Hit {
	type fused struct {
		hit   Hit
		score float64
	}
	merged := map[string]*fused{}
	for i, target := range targets {
		if i >= len(ranked) {
			break
		}
		for rank, h := range toHits(ranked[i]) {
			h.DB = target.Name
			key := h.DB + "|" + h.Path + "|" + fmt.Sprint(h.ChunkIndex)
			contribution := 1.0 / float64(60+rank+1)
			if cur, ok := merged[key]; ok {
				cur.score += contribution
				continue
			}
			merged[key] = &fused{hit: h, score: contribution}
		}
	}
	keys := make([]string, 0, len(merged))
	for k := range merged {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return merged[keys[i]].score > merged[keys[j]].score })
	out := make([]Hit, 0, len(keys))
	for _, k := range keys {
		h := merged[k].hit
		h.Score = merged[k].score
		out = append(out, h)
	}
	return out
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

// Status reports the live state. The health probe runs after releasing the
// lock: a slow Qdrant must not block a model load.
func (e *Engine) Status(ctx context.Context) Status {
	e.mu.Lock()
	st := Status{Collection: e.cfg.Index.Collection, Model: e.cfg.Defaults.Embedding}
	providers := map[string]bool{}
	for k, v := range e.embedders {
		st.Embedders = append(st.Embedders, k)
		providers[v.Provider()] = true
	}
	switch len(providers) {
	case 1:
		for provider := range providers {
			st.Provider = provider
		}
	case 0:
		// No embedder loaded yet.
	default:
		// Several models can be resident at once; a single top-level provider
		// would be whichever map entry won the iteration.
		st.Provider = "mixed"
	}
	for k := range e.rerankers {
		st.Rerankers = append(st.Rerankers, k)
	}
	for _, state := range e.states {
		st.Models = append(st.Models, *state)
	}
	e.mu.Unlock()
	sort.Strings(st.Embedders)
	sort.Strings(st.Rerankers)
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

// LoadModel preloads one catalogue entry. For embeddings it loads every
// dimension used by a registered base, so a later `search --db` does not pay
// the session cost; without a base it uses the model's effective dimensions.
func (e *Engine) LoadModel(ctx context.Context, id string) error {
	e.use.Lock()
	defer e.use.Unlock()
	m, err := e.catalog.Get(id)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	switch m.Task {
	case registry.TaskEmbedding:
		dims := e.dimsForModel(id, m)
		for _, dim := range dims {
			if _, err := e.embedderFor(Target{ModelID: id, Dims: dim}); err != nil {
				return err
			}
		}
		return nil
	case registry.TaskReranker:
		_, err := e.rerankerForID(id)
		return err
	default:
		return fmt.Errorf("%w: %s no es embedding ni reranker", ErrInvalidRequest, id)
	}
}

// UnloadModel closes only the sessions of one model. It is idempotent when the
// model is not resident.
func (e *Engine) UnloadModel(id string) error {
	e.use.Lock()
	defer e.use.Unlock()
	m, err := e.catalog.Get(id)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if m.Task == registry.TaskEmbedding {
		prefix := "embed:" + id + "/"
		for key, emb := range e.embedders {
			if !strings.HasPrefix(key, prefix) {
				continue
			}
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
	} else {
		key := "rerank:" + id
		if rr, ok := e.rerankers[id]; ok {
			if c, ok := rr.(interface{ Close() error }); ok {
				_ = c.Close()
			}
			delete(e.rerankers, id)
			if st, ok := e.states[key]; ok {
				st.State = "not_loaded"
				st.Provider = ""
				st.LoadMS = 0
			}
		}
	}
	e.metrics.Set("argos_models_loaded", float64(len(e.embedders)), metrics.Labels{"kind": "embed"})
	e.metrics.Set("argos_models_loaded", float64(len(e.rerankers)), metrics.Labels{"kind": "rerank"})
	e.metrics.Inc("argos_model_unload_total", metrics.Labels{"model": id})
	trimHeap()
	return nil
}

func (e *Engine) dimsForModel(id string, m registry.Model) []int {
	seen := map[int]bool{}
	if list, err := e.dbs.List(); err == nil {
		for _, meta := range list {
			if meta.Model == id && meta.Dims > 0 {
				seen[meta.Dims] = true
			}
		}
	}
	if id == e.cfg.Defaults.Embedding {
		if e.cfg.Index.Dims > 0 {
			seen[e.cfg.Index.Dims] = true
		} else if d := m.EffectiveDims(); d > 0 {
			seen[d] = true
		}
	}
	if len(seen) == 0 {
		if d := m.EffectiveDims(); d > 0 {
			seen[d] = true
		}
	}
	out := make([]int, 0, len(seen))
	for d := range seen {
		out = append(out, d)
	}
	sort.Ints(out)
	return out
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
		return m.EffectiveDims()
	}
	return 0
}

func (e *Engine) embedderFor(t Target) (Embedder, error) {
	key := fmt.Sprintf("embed:%s/%d", t.ModelID, t.Dims)
	e.mu.Lock()
	if emb, ok := e.embedders[key]; ok {
		e.mu.Unlock()
		return emb, nil
	}
	e.mu.Unlock()

	// Slow path: one loader per model/dims. Different models can load
	// concurrently, and Status keeps reporting "loading" while this runs.
	lk := e.loadLockFor(key)
	lk.Lock()
	defer lk.Unlock()

	e.mu.Lock()
	if emb, ok := e.embedders[key]; ok {
		e.mu.Unlock()
		return emb, nil
	}
	state := &ModelStatus{Key: key, State: "loading"}
	e.states[key] = state
	e.mu.Unlock()
	start := time.Now()

	m, err := e.catalog.Get(t.ModelID)
	if err == nil {
		err = e.checkRAM(m)
	}
	var emb Embedder
	if err == nil {
		emb, err = e.fact.Embedder(m, t.Dims)
		if err == nil && emb == nil {
			err = fmt.Errorf("modelo %s sin fábrica", t.ModelID)
		}
	}

	e.mu.Lock()
	if err == nil {
		e.embedders[key] = emb
		state.State = "ready"
		state.Provider = emb.Provider()
		state.LoadMS = time.Since(start).Milliseconds()
	} else {
		state.State = "error"
		state.Error = err.Error()
	}
	loaded := len(e.embedders)
	e.mu.Unlock()

	if err != nil {
		e.metrics.Inc("argos_model_load_total", metrics.Labels{"model": t.ModelID, "state": "error"})
		return nil, fmt.Errorf("%w: %v", ErrModelUnavailable, err)
	}
	e.metrics.Observe("argos_model_load_seconds", time.Since(start).Seconds(), metrics.Labels{"model": t.ModelID})
	e.metrics.Inc("argos_model_load_total", metrics.Labels{"model": t.ModelID, "state": "ready"})
	e.metrics.Set("argos_models_loaded", float64(loaded), metrics.Labels{"kind": "embed"})
	return emb, nil
}

func (e *Engine) rerankerFor() (Reranker, error) {
	return e.rerankerForID(e.cfg.Defaults.Reranker)
}

func (e *Engine) rerankerForID(id string) (Reranker, error) {
	key := "rerank:" + id
	e.mu.Lock()
	if rr, ok := e.rerankers[id]; ok {
		e.mu.Unlock()
		return rr, nil
	}
	e.mu.Unlock()

	lk := e.loadLockFor(key)
	lk.Lock()
	defer lk.Unlock()

	e.mu.Lock()
	if rr, ok := e.rerankers[id]; ok {
		e.mu.Unlock()
		return rr, nil
	}
	state := &ModelStatus{Key: key, State: "loading"}
	e.states[key] = state
	e.mu.Unlock()
	start := time.Now()

	m, err := e.catalog.Get(id)
	if err == nil {
		err = e.checkRAM(m)
	}
	var rr Reranker
	if err == nil {
		rr, err = e.fact.Reranker(m)
		if err == nil && rr == nil {
			err = fmt.Errorf("reranker %s sin fábrica", id)
		}
	}

	e.mu.Lock()
	if err == nil {
		e.rerankers[id] = rr
		state.State = "ready"
		state.Provider = rr.Provider()
		state.LoadMS = time.Since(start).Milliseconds()
	} else {
		state.State = "error"
		state.Error = err.Error()
	}
	loaded := len(e.rerankers)
	e.mu.Unlock()

	if err != nil {
		e.metrics.Inc("argos_model_load_total", metrics.Labels{"model": id, "state": "error"})
		return nil, fmt.Errorf("%w: %v", ErrModelUnavailable, err)
	}
	e.metrics.Observe("argos_model_load_seconds", time.Since(start).Seconds(), metrics.Labels{"model": id})
	e.metrics.Inc("argos_model_load_total", metrics.Labels{"model": id, "state": "ready"})
	e.metrics.Set("argos_models_loaded", float64(loaded), metrics.Labels{"kind": "rerank"})
	return rr, nil
}

func (e *Engine) loadLockFor(key string) *sync.Mutex {
	e.mu.Lock()
	defer e.mu.Unlock()
	lk := e.loadLocks[key]
	if lk == nil {
		lk = &sync.Mutex{}
		e.loadLocks[key] = lk
	}
	return lk
}

func (e *Engine) checkRAM(m registry.Model) error {
	if m.Specs.RAMMB <= 0 {
		return nil
	}
	available := 0
	if e.ramAvailable != nil {
		available = e.ramAvailable()
	}
	if available > 0 && m.Specs.RAMMB+ramHeadroomMB > available {
		return fmt.Errorf("RAM insuficiente para %s: necesita %d MB + %d MB de margen y hay %d MB",
			m.ID, m.Specs.RAMMB, ramHeadroomMB, available)
	}
	return nil
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
