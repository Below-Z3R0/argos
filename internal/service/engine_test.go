package service

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Below-Z3R0/argos/internal/config"
	dbpkg "github.com/Below-Z3R0/argos/internal/db"
	"github.com/Below-Z3R0/argos/internal/registry"
	"github.com/Below-Z3R0/argos/internal/types"
)

type fakeEmbedder struct {
	calls int
	prov  string
}

func (f *fakeEmbedder) EmbedQuery(context.Context, string) ([]float32, error) {
	f.calls++
	return []float32{1, 0}, nil
}
func (f *fakeEmbedder) Dim() int { return 2 }
func (f *fakeEmbedder) Provider() string {
	if f.prov == "" {
		return "CPU"
	}
	return f.prov
}

type fakeStore struct {
	hits        []types.Hit
	searchCalls int
	hybridCalls int
}

func (f *fakeStore) Search(context.Context, string, []float32, int, map[string]string) ([]types.Hit, error) {
	f.searchCalls++
	return f.hits, nil
}

func (f *fakeStore) SearchHybrid(context.Context, string, []float32, []uint32, []float32, int, map[string]string) ([]types.Hit, error) {
	f.hybridCalls++
	return f.hits, nil
}

func (f *fakeStore) Health(context.Context) error { return nil }

type fakeReranker struct{}

func (fakeReranker) Score(_ context.Context, _ string, passages []string) ([]float32, error) {
	out := make([]float32, len(passages))
	for i := range passages {
		out[i] = float32(i+1) / float32(len(passages))
	}
	return out, nil
}
func (fakeReranker) Provider() string { return "CPU" }

func catalog() *registry.Catalog {
	return &registry.Catalog{Models: []registry.Model{
		{ID: "embed", Name: "E", Task: registry.TaskEmbedding, Tier: "balanced", Default: true, Specs: registry.Specs{Dims: 2}},
		{ID: "rr", Name: "R", Task: registry.TaskReranker, Tier: "balanced", Default: true},
	}}
}

func hit(path, text string, score float32, index int) types.Hit {
	return types.Hit{Score: score, Payload: map[string]any{
		"path": path, "text": text, "heading": "H", "chunk_index": int64(index),
	}}
}

func testConfig(t *testing.T) config.Config {
	cfg := config.Default()
	cfg.Paths.Home = t.TempDir()
	cfg.Defaults.Embedding = "embed"
	cfg.Defaults.Reranker = "rr"
	cfg.Index.Collection = "c"
	cfg.Index.Dims = 2
	return cfg
}

func TestSearchUsesDefaults(t *testing.T) {
	cfg := testConfig(t)
	emb := &fakeEmbedder{}
	e := New(cfg, catalog(), &fakeStore{hits: []types.Hit{hit("a.md", "uno", 0.9, 1), hit("a.md", "dos", 0.8, 2), hit("a.md", "tres", 0.7, 3)}}, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) { return emb, nil },
	})
	resp, err := e.Search(context.Background(), SearchRequest{Query: "q", K: 2, Rerank: "off"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Hits) != 2 {
		t.Fatalf("hits %d", len(resp.Hits))
	}
	if resp.Collection != "c" || resp.Model != "embed" {
		t.Fatalf("resp %+v", resp)
	}
	if emb.calls != 1 {
		t.Fatalf("embed calls %d", emb.calls)
	}
}

func TestRerankerReordersAndLoadsOnce(t *testing.T) {
	cfg := testConfig(t)
	e := New(cfg, catalog(), &fakeStore{hits: []types.Hit{hit("a.md", "malo", 0.9, 4), hit("b.md", "bueno", 0.5, 5)}}, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) { return &fakeEmbedder{}, nil },
		Reranker: func(registry.Model) (Reranker, error) { return fakeReranker{}, nil },
	})
	resp, err := e.Search(context.Background(), SearchRequest{Query: "q", K: 2})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Hits[0].Text != "malo" && resp.Hits[0].Text != "bueno" {
		t.Fatalf("hits %+v", resp.Hits)
	}
	// The fake scores by position, so the last passage wins.
	if resp.Hits[0].Text != "bueno" {
		t.Fatalf("reranker not applied: %+v", resp.Hits)
	}
	st := e.Status(context.Background())
	if len(st.Embedders) != 1 || len(st.Rerankers) != 1 {
		t.Fatalf("status %+v", st)
	}
}

func TestRerankOnWithoutModelFails(t *testing.T) {
	cfg := testConfig(t)
	e := New(cfg, catalog(), &fakeStore{hits: []types.Hit{hit("a.md", "x", 0.9, 6)}}, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) { return &fakeEmbedder{}, nil },
		Reranker: func(registry.Model) (Reranker, error) { return nil, errors.New("no instalado") },
	})
	if _, err := e.Search(context.Background(), SearchRequest{Query: "q", Rerank: "on"}); err == nil {
		t.Fatal("expected error")
	}
}

func TestSearchModeSelectsStorePath(t *testing.T) {
	cfg := testConfig(t)
	store := &fakeStore{hits: []types.Hit{hit("a.md", "x", 0.9, 1)}}
	e := New(cfg, catalog(), store, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) { return &fakeEmbedder{}, nil },
	})

	if _, err := e.Search(context.Background(), SearchRequest{Query: "q", Mode: "dense", Rerank: "off"}); err != nil {
		t.Fatal(err)
	}
	if store.searchCalls != 1 || store.hybridCalls != 0 {
		t.Fatalf("dense: search=%d hybrid=%d", store.searchCalls, store.hybridCalls)
	}

	if _, err := e.Search(context.Background(), SearchRequest{Query: "q", Mode: "hybrid", Rerank: "off"}); err != nil {
		t.Fatal(err)
	}
	if store.searchCalls != 1 || store.hybridCalls != 1 {
		t.Fatalf("hybrid: search=%d hybrid=%d", store.searchCalls, store.hybridCalls)
	}

	if _, err := e.Search(context.Background(), SearchRequest{Query: "q", Mode: "bm25"}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("invalid mode error = %v, want ErrInvalidRequest", err)
	}
}

func TestSearchEmbedsOncePerModelAcrossDBs(t *testing.T) {
	cfg := testConfig(t)
	src := t.TempDir()
	mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
	for _, name := range []string{"uno", "dos"} {
		dir := filepath.Join(src, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if _, err := mgr.Create(dbpkg.Meta{Name: name, Source: dir, Model: "embed", Dims: 2}); err != nil {
			t.Fatal(err)
		}
	}
	emb := &fakeEmbedder{}
	e := New(cfg, catalog(), &fakeStore{hits: []types.Hit{hit("a.md", "x", 0.9, 1)}}, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) { return emb, nil },
	})
	if _, err := e.Search(context.Background(), SearchRequest{Query: "q", DB: "all", Rerank: "off"}); err != nil {
		t.Fatal(err)
	}
	if emb.calls != 1 {
		t.Fatalf("query embedded %d times, want 1 for two bases sharing a model", emb.calls)
	}
}

func TestSearchDifferentModelsLoadsEachOnce(t *testing.T) {
	cfg := testConfig(t)
	cat := &registry.Catalog{Models: []registry.Model{
		{ID: "embed-a", Name: "A", Task: registry.TaskEmbedding, Tier: "balanced", Default: true, Specs: registry.Specs{Dims: 2}},
		{ID: "embed-b", Name: "B", Task: registry.TaskEmbedding, Tier: "fast", Specs: registry.Specs{Dims: 3}},
	}}
	src := t.TempDir()
	mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
	for _, item := range []struct {
		name  string
		model string
		dims  int
	}{{"uno", "embed-a", 2}, {"dos", "embed-b", 3}} {
		dir := filepath.Join(src, item.name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if _, err := mgr.Create(dbpkg.Meta{Name: item.name, Source: dir, Model: item.model, Dims: item.dims}); err != nil {
			t.Fatal(err)
		}
	}
	byModel := map[string]*fakeEmbedder{
		"embed-a": {prov: "CPU"},
		"embed-b": {prov: "CUDA"},
	}
	e := New(cfg, cat, &fakeStore{hits: []types.Hit{hit("a.md", "x", 0.9, 1)}}, Factories{
		Embedder: func(m registry.Model, _ int) (Embedder, error) { return byModel[m.ID], nil },
	})
	if _, err := e.Search(context.Background(), SearchRequest{Query: "q", DB: "all", Rerank: "off"}); err != nil {
		t.Fatal(err)
	}
	if byModel["embed-a"].calls != 1 || byModel["embed-b"].calls != 1 {
		t.Fatalf("calls: a=%d b=%d", byModel["embed-a"].calls, byModel["embed-b"].calls)
	}
	st := e.Status(context.Background())
	if st.Provider != "mixed" {
		t.Fatalf("provider = %q, want mixed", st.Provider)
	}
	if len(st.Embedders) != 2 {
		t.Fatalf("embedders = %v", st.Embedders)
	}
}

func TestLoadModelAndUnloadModel(t *testing.T) {
	cfg := testConfig(t)
	loads := 0
	e := New(cfg, catalog(), &fakeStore{}, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) {
			loads++
			return &fakeEmbedder{}, nil
		},
	})
	if err := e.LoadModel(context.Background(), "embed"); err != nil {
		t.Fatal(err)
	}
	if loads != 1 || len(e.Status(context.Background()).Embedders) != 1 {
		t.Fatalf("loads=%d status=%+v", loads, e.Status(context.Background()))
	}
	if err := e.UnloadModel("embed"); err != nil {
		t.Fatal(err)
	}
	if loads != 1 || len(e.Status(context.Background()).Embedders) != 0 {
		t.Fatalf("unload left state: loads=%d status=%+v", loads, e.Status(context.Background()))
	}
	if err := e.LoadModel(context.Background(), "embed"); err != nil {
		t.Fatal(err)
	}
	if loads != 2 {
		t.Fatalf("second load did not create a new session: %d", loads)
	}
}

func TestLoadModelRejectsWhenRAMIsTight(t *testing.T) {
	cfg := testConfig(t)
	cat := &registry.Catalog{Models: []registry.Model{
		{ID: "embed", Name: "E", Task: registry.TaskEmbedding, Tier: "balanced", Default: true,
			Specs: registry.Specs{Dims: 2, RAMMB: 1024}},
	}}
	e := New(cfg, cat, &fakeStore{}, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) { return &fakeEmbedder{}, nil },
	})
	e.ramAvailable = func() int { return 512 }
	if err := e.LoadModel(context.Background(), "embed"); err == nil {
		t.Fatal("expected a RAM error")
	}
	st := e.Status(context.Background())
	if len(st.Models) != 1 || st.Models[0].State != "error" {
		t.Fatalf("state after RAM error: %+v", st.Models)
	}
}

func TestStatusDoesNotBlockWhileModelLoads(t *testing.T) {
	cfg := testConfig(t)
	started := make(chan struct{})
	release := make(chan struct{})
	e := New(cfg, catalog(), &fakeStore{}, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) {
			close(started)
			<-release
			return &fakeEmbedder{}, nil
		},
	})
	errCh := make(chan error, 1)
	go func() { errCh <- e.LoadModel(context.Background(), "embed") }()
	<-started

	statusCh := make(chan string, 1)
	go func() {
		st := e.Status(context.Background())
		for _, m := range st.Models {
			if m.Key == "embed:embed/2" {
				statusCh <- m.State
				return
			}
		}
		statusCh <- ""
	}()
	select {
	case state := <-statusCh:
		if state != "loading" {
			t.Fatalf("state = %q, want loading", state)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Status blocked while the model was loading")
	}
	close(release)
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
}

func TestTargetsSelector(t *testing.T) {
	cfg := testConfig(t)
	src := t.TempDir()
	mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
	for _, name := range []string{"uno", "dos"} {
		dir := filepath.Join(src, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if _, err := mgr.Create(dbpkg.Meta{Name: name, Source: dir, Model: "embed", Dims: 2}); err != nil {
			t.Fatal(err)
		}
	}
	e := New(cfg, catalog(), &fakeStore{}, Factories{})
	check := func(req SearchRequest, want ...string) {
		t.Helper()
		got, err := e.targets(req)
		if err != nil {
			t.Fatalf("%+v: %v", req, err)
		}
		if len(got) != len(want) {
			t.Fatalf("%+v: got %d targets, want %d", req, len(got), len(want))
		}
		for i := range want {
			if got[i].Name != want[i] {
				t.Fatalf("%+v: target %d = %q, want %q", req, i, got[i].Name, want[i])
			}
		}
	}
	check(SearchRequest{DB: "all"}, "dos", "uno")
	check(SearchRequest{DB: "uno"}, "uno")
	check(SearchRequest{DB: "uno,dos"}, "dos", "uno")
	check(SearchRequest{DB: "all", ExcludeDBs: []string{"uno"}}, "dos")
	check(SearchRequest{DB: "all", Model: "embed"}, "dos", "uno")
	if _, err := e.targets(SearchRequest{DB: "all", Model: "nope"}); err == nil {
		t.Fatal("expected an error for a model with no bases")
	}
}

func TestResolveNamedDB(t *testing.T) {
	cfg := testConfig(t)
	src := t.TempDir()
	mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
	meta, err := mgr.Create(dbpkg.Meta{Name: "notas", Source: src, Model: "embed", Dims: 2})
	if err != nil {
		t.Fatal(err)
	}
	e := New(cfg, catalog(), &fakeStore{}, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) { return &fakeEmbedder{}, nil },
	})
	target, err := e.Resolve("notas")
	if err != nil {
		t.Fatal(err)
	}
	if target.Collection != meta.Collection || target.ModelID != "embed" {
		t.Fatalf("target %+v", target)
	}
}

func TestRejectsEmptyQuery(t *testing.T) {
	e := New(testConfig(t), catalog(), &fakeStore{}, Factories{})
	if _, err := e.Search(context.Background(), SearchRequest{Query: "  "}); err == nil {
		t.Fatal("expected error")
	}
}
