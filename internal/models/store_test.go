package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/Below-Z3R0/argos/internal/registry"
)

func serverWith(t *testing.T, body []byte) (*httptest.Server, string) {
	t.Helper()
	sum := sha256.Sum256(body)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv, hex.EncodeToString(sum[:])
}

func modelFor(url, sha string) registry.Model {
	return registry.Model{
		ID: "m", Name: "M", Task: registry.TaskEmbedding, Tier: "balanced",
		Files: []registry.File{{URL: url, Dest: "model.onnx", SizeMB: 1, SHA256: sha}},
	}
}

func TestInstallVerifiesAndCommits(t *testing.T) {
	body := []byte("weights")
	srv, sum := serverWith(t, body)
	store := New(t.TempDir())

	if err := store.Install(context.Background(), modelFor(srv.URL, sum), Options{}); err != nil {
		t.Fatalf("install: %v", err)
	}
	rec, ok := store.Status(registry.TaskEmbedding, "m")
	if !ok {
		t.Fatal("model should be installed")
	}
	if rec.SizeBytes != int64(len(body)) {
		t.Fatalf("size %d", rec.SizeBytes)
	}
	dir, _ := store.Dir(registry.TaskEmbedding, "m")
	if _, err := os.Stat(filepath.Join(dir, "model.onnx")); err != nil {
		t.Fatalf("file missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, ".tmp")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("temp dir must be removed")
	}
}

func TestInstallRejectsHashMismatch(t *testing.T) {
	body := []byte("weights")
	srv, _ := serverWith(t, body)
	store := New(t.TempDir())
	wrong := "0000000000000000000000000000000000000000000000000000000000000000"

	err := store.Install(context.Background(), modelFor(srv.URL, wrong), Options{})
	if err == nil {
		t.Fatal("expected hash mismatch")
	}
	if _, ok := store.Status(registry.TaskEmbedding, "m"); ok {
		t.Fatal("failed install must not look installed")
	}
	dir, _ := store.Dir(registry.TaskEmbedding, "m")
	if _, err := os.Stat(filepath.Join(dir, "model.onnx")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("failed install must not leave the artifact")
	}
}

func TestInstallRefusesUnpinned(t *testing.T) {
	srv, _ := serverWith(t, []byte("x"))
	store := New(t.TempDir())
	err := store.Install(context.Background(), modelFor(srv.URL, ""), Options{})
	if !errors.Is(err, ErrUnpinned) {
		t.Fatalf("expected ErrUnpinned, got %v", err)
	}
}

func TestInstallAllowUnpinned(t *testing.T) {
	srv, _ := serverWith(t, []byte("x"))
	store := New(t.TempDir())
	if err := store.Install(context.Background(), modelFor(srv.URL, ""), Options{AllowUnpinned: true}); err != nil {
		t.Fatalf("install: %v", err)
	}
	if _, ok := store.Status(registry.TaskEmbedding, "m"); !ok {
		t.Fatal("should be installed")
	}
}

func TestVerifyAndRemove(t *testing.T) {
	srv, sum := serverWith(t, []byte("weights"))
	store := New(t.TempDir())
	m := modelFor(srv.URL, sum)
	if err := store.Install(context.Background(), m, Options{}); err != nil {
		t.Fatal(err)
	}
	if err := store.Verify(m.Task, m.ID, m.Files); err != nil {
		t.Fatalf("verify: %v", err)
	}
	if err := store.Remove(m.Task, m.ID); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if _, ok := store.Status(m.Task, m.ID); ok {
		t.Fatal("still installed after remove")
	}
}

func TestRemoveMissing(t *testing.T) {
	store := New(t.TempDir())
	err := store.Remove(registry.TaskReranker, "nope")
	if !errors.Is(err, ErrNotInstalled) {
		t.Fatalf("expected ErrNotInstalled, got %v", err)
	}
}
