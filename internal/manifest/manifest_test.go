package manifest

import (
	"path/filepath"
	"testing"
)

func TestPutGetAllDelete(t *testing.T) {
	m, err := Open(filepath.Join(t.TempDir(), "manifest.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if _, ok, err := m.Get("a.md"); err != nil || ok {
		t.Fatalf("empty manifest: ok=%v err=%v", ok, err)
	}
	state := FileState{Rel: "a.md", Hash: "h1", Chunks: []ChunkRef{{ID: "1", Hash: "c1"}, {ID: "2", Hash: "c2"}}}
	if err := m.Put(state); err != nil {
		t.Fatal(err)
	}
	got, ok, err := m.Get("a.md")
	if err != nil || !ok {
		t.Fatalf("get: ok=%v err=%v", ok, err)
	}
	if got.Hash != "h1" || len(got.Chunks) != 2 {
		t.Fatalf("state %+v", got)
	}
	all, err := m.All()
	if err != nil || len(all) != 1 {
		t.Fatalf("all: %v %v", all, err)
	}
	if err := m.Delete("a.md"); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := m.Get("a.md"); ok {
		t.Fatal("still present after delete")
	}
}

func TestReopenKeepsData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.db")
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Put(FileState{Rel: "b.md", Hash: "h2"}); err != nil {
		t.Fatal(err)
	}
	m.Close()

	m2, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()
	if _, ok, _ := m2.Get("b.md"); !ok {
		t.Fatal("data lost after reopen")
	}
}
