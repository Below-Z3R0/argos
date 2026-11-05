package search

import (
	"context"
	"testing"

	"github.com/Below-Z3R0/argos/internal/types"
)

type fakeEmbedder struct{ calls int }

func (f *fakeEmbedder) EmbedQuery(context.Context, string) ([]float32, error) {
	f.calls++
	return []float32{1, 0}, nil
}

type fakeStore struct{ hits []types.Hit }

func (f *fakeStore) Search(_ context.Context, _ string, _ []float32, limit int, _ map[string]string) ([]types.Hit, error) {
	out := f.hits
	if limit < len(out) {
		out = out[:limit]
	}
	return out, nil
}

type fakeReranker struct{ scores map[string]float32 }

func (f *fakeReranker) Score(_ context.Context, _ string, passages []string) ([]float32, error) {
	out := make([]float32, len(passages))
	for i, p := range passages {
		out[i] = f.scores[p]
	}
	return out, nil
}

func hit(path, text string, score float32) types.Hit {
	return types.Hit{ID: path + text, Score: score, Payload: map[string]any{
		"path": path, "heading": "H", "text": text, "tokens": int64(10), "chunk_index": int64(0),
	}}
}

func TestLimitsAndPerFile(t *testing.T) {
	store := &fakeStore{hits: []types.Hit{
		hit("a.md", "a1", 0.9), hit("a.md", "a2", 0.8), hit("a.md", "a3", 0.7),
		hit("b.md", "b1", 0.6),
	}}
	s := Searcher{Embedder: &fakeEmbedder{}, Store: store, Collection: "c", K: 3, MaxPerFile: 2}
	got, err := s.Search(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("results %d", len(got))
	}
	perFile := map[string]int{}
	for _, r := range got {
		perFile[r.Path]++
	}
	if perFile["a.md"] > 2 {
		t.Fatalf("per-file cap ignored: %+v", perFile)
	}
}

func TestMinScoreFilters(t *testing.T) {
	store := &fakeStore{hits: []types.Hit{hit("a.md", "a", 0.9), hit("b.md", "b", 0.4)}}
	s := Searcher{Embedder: &fakeEmbedder{}, Store: store, Collection: "c", K: 5, MinScore: 0.5}
	got, _ := s.Search(context.Background(), "q")
	if len(got) != 1 || got[0].Path != "a.md" {
		t.Fatalf("got %+v", got)
	}
}

func TestRerankerReorders(t *testing.T) {
	store := &fakeStore{hits: []types.Hit{hit("a.md", "malo", 0.9), hit("b.md", "bueno", 0.5)}}
	rr := &fakeReranker{scores: map[string]float32{"malo": 0.1, "bueno": 0.95}}
	s := Searcher{Embedder: &fakeEmbedder{}, Store: store, Reranker: rr, Collection: "c", K: 2}
	got, err := s.Search(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Text != "bueno" {
		t.Fatalf("reranker did not reorder: %+v", got)
	}
}

func TestRejectsEmptyQuery(t *testing.T) {
	s := Searcher{Embedder: &fakeEmbedder{}, Store: &fakeStore{}}
	if _, err := s.Search(context.Background(), "  "); err == nil {
		t.Fatal("expected error")
	}
}
