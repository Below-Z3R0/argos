package ingest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/Below-Z3R0/argos/internal/chunk"
	"github.com/Below-Z3R0/argos/internal/manifest"
	"github.com/Below-Z3R0/argos/internal/parse"
	"github.com/Below-Z3R0/argos/internal/types"
	"github.com/Below-Z3R0/argos/internal/vault"
)

type fakeEmbedder struct {
	dim   int
	calls int
	texts int
	fail  bool
}

func (f *fakeEmbedder) Name() string { return "fake" }
func (f *fakeEmbedder) Dim() int     { return f.dim }
func (f *fakeEmbedder) EmbedPassages(_ context.Context, texts []string) ([][]float32, error) {
	if f.fail {
		return nil, errors.New("embedder down")
	}
	f.calls++
	f.texts += len(texts)
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, f.dim)
		v[0] = float32(len(t) % 100)
		v[1] = float32(i)
		out[i] = v
	}
	return out, nil
}

type fakeStore struct {
	points  map[string]types.Point
	deletes int
	upserts int
}

func newFakeStore() *fakeStore { return &fakeStore{points: map[string]types.Point{}} }

func (f *fakeStore) EnsureCollection(context.Context, string, int, string) (bool, error) {
	return true, nil
}
func (f *fakeStore) EnsurePayloadIndexes(context.Context, string) error { return nil }
func (f *fakeStore) UpsertWithWait(_ context.Context, _ string, points []types.Point, _ bool) error {
	for _, p := range points {
		f.points[p.ID] = p
	}
	f.upserts++
	return nil
}
func (f *fakeStore) DeleteByIDs(_ context.Context, _ string, ids []string) error {
	for _, id := range ids {
		delete(f.points, id)
	}
	f.deletes++
	return nil
}
func (f *fakeStore) DeleteByPayload(_ context.Context, _, field, value string) error {
	for id, p := range f.points {
		if p.Payload[field] == value {
			delete(f.points, id)
		}
	}
	return nil
}

func fixture(t *testing.T) (string, []vault.File, *manifest.Manifest) {
	t.Helper()
	root := t.TempDir()
	write := func(name, body string) {
		p := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("a.md", "# A\n\n"+strings.Repeat("Texto de la nota A. ", 20)+"\n\n## Otra\n\nMás texto A.\n")
	write("sub/b.md", "# B\n\n"+strings.Repeat("Texto de la nota B. ", 20)+"\n")

	files, err := vault.Walk(root, vault.Options{Include: []string{".md"}})
	if err != nil {
		t.Fatal(err)
	}
	man, err := manifest.Open(filepath.Join(t.TempDir(), "m.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { man.Close() })
	return root, files, man
}

func splitter() chunk.Splitter {
	return chunk.Splitter{
		Counter:   func(s string) int { return len(strings.Fields(s)) },
		MaxTokens: 60,
		MinTokens: 0,
		Namespace: uuid.NameSpaceURL,
	}
}

// A full reindex used to delete by path before embedding. If the model failed,
// the file lost every point. This locks the safe order: embed first, delete
// only after the new vectors are stored.
func TestFullReindexKeepsPointsWhenEmbeddingFails(t *testing.T) {
	_, files, man := fixture(t)
	store := newFakeStore()
	first := &fakeEmbedder{dim: 4}
	stats, err := Run(context.Background(), files, parse.DefaultRegistry(), splitter(), first, store, man, Options{Collection: "c", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	before := len(store.points)
	if before == 0 || before != stats.Embed {
		t.Fatalf("first run stored %d points for %d chunks", before, stats.Embed)
	}

	broken := &fakeEmbedder{dim: 4, fail: true}
	failed, err := Run(context.Background(), files, parse.DefaultRegistry(), splitter(), broken, store, man, Options{Collection: "c", Model: "m", Full: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(failed.Errors) == 0 {
		t.Fatal("expected the embedding failure to be recorded per file")
	}
	if len(store.points) != before {
		t.Fatalf("failed full reindex left %d points, want the previous %d", len(store.points), before)
	}
}

func TestFirstRunIndexesEverything(t *testing.T) {
	_, files, man := fixture(t)
	emb := &fakeEmbedder{dim: 4}
	store := newFakeStore()
	stats, err := Run(context.Background(), files, parse.DefaultRegistry(), splitter(), emb, store, man, Options{Collection: "c", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Added != 2 || stats.Embed == 0 {
		t.Fatalf("stats %+v", stats)
	}
	if len(store.points) != stats.Embed {
		t.Fatalf("points %d vs embedded %d", len(store.points), stats.Embed)
	}
	for _, p := range store.points {
		if p.Payload["text"] == nil || p.Payload["file_hash"] == nil {
			t.Fatalf("payload incomplete: %+v", p.Payload)
		}
	}
}

func TestSecondRunSkipsEverything(t *testing.T) {
	_, files, man := fixture(t)
	emb := &fakeEmbedder{dim: 4}
	store := newFakeStore()
	opts := Options{Collection: "c", Model: "m"}
	if _, err := Run(context.Background(), files, parse.DefaultRegistry(), splitter(), emb, store, man, opts); err != nil {
		t.Fatal(err)
	}
	calls := emb.calls
	stats, err := Run(context.Background(), files, parse.DefaultRegistry(), splitter(), emb, store, man, opts)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Skipped != 2 || stats.Embed != 0 {
		t.Fatalf("stats %+v", stats)
	}
	if emb.calls != calls {
		t.Fatal("no debería haber embedido nada")
	}
}

func TestChangedFileReusesUnchangedChunks(t *testing.T) {
	root, files, man := fixture(t)
	emb := &fakeEmbedder{dim: 4}
	store := newFakeStore()
	opts := Options{Collection: "c", Model: "m"}
	if _, err := Run(context.Background(), files, parse.DefaultRegistry(), splitter(), emb, store, man, opts); err != nil {
		t.Fatal(err)
	}

	// Append a new section: the chunks before it must keep their ids.
	path := filepath.Join(root, "a.md")
	data, _ := os.ReadFile(path)
	os.WriteFile(path, append(data, []byte("\n## Nueva\n\nSección agregada.\n")...), 0o644)
	files, _ = vault.Walk(root, vault.Options{Include: []string{".md"}})

	stats, err := Run(context.Background(), files, parse.DefaultRegistry(), splitter(), emb, store, man, opts)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Updated != 1 {
		t.Fatalf("stats %+v", stats)
	}
	if stats.Reused == 0 {
		t.Fatalf("expected reused chunks, got %+v", stats)
	}
	if stats.Embed == 0 {
		t.Fatalf("expected new chunks for the added section, got %+v", stats)
	}
}

func TestDeletedFileIsPruned(t *testing.T) {
	root, files, man := fixture(t)
	emb := &fakeEmbedder{dim: 4}
	store := newFakeStore()
	opts := Options{Collection: "c", Model: "m"}
	if _, err := Run(context.Background(), files, parse.DefaultRegistry(), splitter(), emb, store, man, opts); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(root, "sub", "b.md")); err != nil {
		t.Fatal(err)
	}
	remaining, _ := vault.Walk(root, vault.Options{Include: []string{".md"}})
	stats, err := Run(context.Background(), remaining, parse.DefaultRegistry(), splitter(), emb, store, man, opts)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Removed != 1 {
		t.Fatalf("stats %+v", stats)
	}
	for _, p := range store.points {
		if p.Payload["path"] == "sub/b.md" {
			t.Fatal("stale point survived")
		}
	}
}

func TestDryRunTouchesNothing(t *testing.T) {
	_, files, man := fixture(t)
	emb := &fakeEmbedder{dim: 4}
	store := newFakeStore()
	stats, err := Run(context.Background(), files, parse.DefaultRegistry(), splitter(), emb, store, man, Options{Collection: "c", Model: "m", DryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Added != 2 || len(store.points) != 0 {
		t.Fatalf("dry run wrote: stats=%+v points=%d", stats, len(store.points))
	}
}
