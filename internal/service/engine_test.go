package service

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/Below-Z3R0/argos/internal/config"
	dbpkg "github.com/Below-Z3R0/argos/internal/db"
	"github.com/Below-Z3R0/argos/internal/registry"
	"github.com/Below-Z3R0/argos/internal/types"
)

type fakeEmbedder struct{ calls int }

func (f *fakeEmbedder) EmbedQuery(context.Context, string) ([]float32, error) {
	f.calls++
	return []float32{1, 0}, nil
}
func (f *fakeEmbedder) Dim() int         { return 2 }
func (f *fakeEmbedder) Provider() string { return "CPU" }

type fakeStore struct{ hits []types.Hit }

func (f *fakeStore) Search(context.Context, string, []float32, int, map[string]string) ([]types.Hit, error) {
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

func hit(path, text string, score float32) types.Hit {
	return types.Hit{Score: score, Payload: map[string]any{"path": path, "text": text, "heading": "H"}}
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
	e := New(cfg, catalog(), &fakeStore{hits: []types.Hit{hit("a.md", "uno", 0.9), hit("a.md", "dos", 0.8), hit("a.md", "tres", 0.7)}}, Factories{
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
	e := New(cfg, catalog(), &fakeStore{hits: []types.Hit{hit("a.md", "malo", 0.9), hit("b.md", "bueno", 0.5)}}, Factories{
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
	e := New(cfg, catalog(), &fakeStore{hits: []types.Hit{hit("a.md", "x", 0.9)}}, Factories{
		Embedder: func(registry.Model, int) (Embedder, error) { return &fakeEmbedder{}, nil },
		Reranker: func(registry.Model) (Reranker, error) { return nil, errors.New("no instalado") },
	})
	if _, err := e.Search(context.Background(), SearchRequest{Query: "q", Rerank: "on"}); err == nil {
		t.Fatal("expected error")
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
