package qdrant

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

func testStore(t *testing.T) *Store {
	t.Helper()
	host := os.Getenv("QDRANT_HOST")
	if host == "" {
		t.Skip("set QDRANT_HOST to run the Qdrant tests")
	}
	port := 6334
	if v := os.Getenv("QDRANT_PORT"); v != "" {
		fmt.Sscan(v, &port)
	}
	s, err := New(host, port)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func collection(t *testing.T) string {
	return fmt.Sprintf("test_%d", time.Now().UnixNano())
}

func vec(x float32) []float32 {
	v := make([]float32, 8)
	v[0] = x
	v[1] = 1 - x
	return v
}

func TestEnsureCollectionIsIdempotent(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	name := collection(t)
	defer s.DeleteCollection(ctx, name)

	for i := 0; i < 2; i++ {
		created, err := s.EnsureCollection(ctx, name, 8, "")
		if err != nil {
			t.Fatalf("attempt %d: %v", i, err)
		}
		if i == 0 && !created {
			t.Fatal("first call did not create the collection")
		}
		if i == 1 && created {
			t.Fatal("second call reported a new collection")
		}
	}
}

func TestUpsertCountAndSearch(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	name := collection(t)
	defer s.DeleteCollection(ctx, name)

	if _, err := s.EnsureCollection(ctx, name, 8, ""); err != nil {
		t.Fatal(err)
	}
	err := s.Upsert(ctx, name, []Point{
		{ID: "11111111-1111-1111-1111-111111111111", Vector: vec(1), Payload: map[string]any{"path": "a.md", "file_hash": "h1"}},
		{ID: "22222222-2222-2222-2222-222222222222", Vector: vec(0), Payload: map[string]any{"path": "b.md", "file_hash": "h2"}},
		{ID: "33333333-3333-3333-3333-333333333333", Vector: vec(0.5), Payload: map[string]any{"path": "c.md", "file_hash": "h1"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	n, err := s.Count(ctx, name, nil)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Fatalf("count %d, want 3", n)
	}

	hits, err := s.Search(ctx, name, vec(1), 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 {
		t.Fatalf("hits %d", len(hits))
	}
	if hits[0].Payload["path"] != "a.md" {
		t.Fatalf("closest hit: %+v", hits[0])
	}
}

func TestUpsertIsIdempotent(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	name := collection(t)
	defer s.DeleteCollection(ctx, name)
	if _, err := s.EnsureCollection(ctx, name, 8, ""); err != nil {
		t.Fatal(err)
	}
	p := []Point{{ID: "44444444-4444-4444-4444-444444444444", Vector: vec(1), Payload: map[string]any{"file_hash": "h"}}}
	for i := 0; i < 2; i++ {
		if err := s.Upsert(ctx, name, p); err != nil {
			t.Fatal(err)
		}
	}
	n, _ := s.Count(ctx, name, nil)
	if n != 1 {
		t.Fatalf("count %d, want 1", n)
	}
}

func TestDeleteByPayload(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	name := collection(t)
	defer s.DeleteCollection(ctx, name)
	if _, err := s.EnsureCollection(ctx, name, 8, ""); err != nil {
		t.Fatal(err)
	}
	err := s.Upsert(ctx, name, []Point{
		{ID: "55555555-5555-5555-5555-555555555555", Vector: vec(1), Payload: map[string]any{"file_hash": "gone"}},
		{ID: "66666666-6666-6666-6666-666666666666", Vector: vec(0.5), Payload: map[string]any{"file_hash": "gone"}},
		{ID: "77777777-7777-7777-7777-777777777777", Vector: vec(0), Payload: map[string]any{"file_hash": "keep"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.DeleteByPayload(ctx, name, "file_hash", "gone"); err != nil {
		t.Fatal(err)
	}
	n, _ := s.Count(ctx, name, nil)
	if n != 1 {
		t.Fatalf("count %d, want 1", n)
	}
	kept, _ := s.Count(ctx, name, map[string]string{"file_hash": "keep"})
	if kept != 1 {
		t.Fatalf("kept %d", kept)
	}
}
