package db

import (
	"os"
	"path/filepath"
	"testing"
)

func manager(t *testing.T) *Manager {
	t.Helper()
	return New(filepath.Join(t.TempDir(), "dbs"))
}

func source(t *testing.T, name string) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestCreateUsesFolderName(t *testing.T) {
	m := manager(t)
	meta, err := m.Create(Meta{Source: source(t, "Mi Carpeta")})
	if err != nil {
		t.Fatal(err)
	}
	if meta.Name != "mi-carpeta" {
		t.Fatalf("name %q", meta.Name)
	}
	if meta.Collection != "argos_mi-carpeta" {
		t.Fatalf("collection %q", meta.Collection)
	}
	if got, err := m.Get(meta.Name); err != nil || got.Source != meta.Source {
		t.Fatalf("get: %+v %v", got, err)
	}
}

func TestCreateCollisionGetsSuffix(t *testing.T) {
	m := manager(t)
	a, err := m.Create(Meta{Source: source(t, "vault")})
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Create(Meta{Source: source(t, "vault")})
	if err != nil {
		t.Fatal(err)
	}
	if a.Name == b.Name {
		t.Fatalf("names collide: %s", a.Name)
	}
	if len(b.Name) <= len("vault-") {
		t.Fatalf("expected a suffix, got %q", b.Name)
	}
}

func TestListAndRemove(t *testing.T) {
	m := manager(t)
	if _, err := m.Create(Meta{Name: "a", Source: source(t, "a")}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Create(Meta{Name: "b", Source: source(t, "b")}); err != nil {
		t.Fatal(err)
	}
	list, err := m.List()
	if err != nil || len(list) != 2 {
		t.Fatalf("list %v %v", list, err)
	}
	if err := m.Remove("a"); err != nil {
		t.Fatal(err)
	}
	list, _ = m.List()
	if len(list) != 1 || list[0].Name != "b" {
		t.Fatalf("after remove: %+v", list)
	}
	if err := m.Remove("nope"); err == nil {
		t.Fatal("expected error removing a missing base")
	}
}

func TestRejectsMissingSource(t *testing.T) {
	m := manager(t)
	if _, err := m.Create(Meta{Source: "/no/existe/esta/ruta"}); err == nil {
		t.Fatal("expected error")
	}
}

func TestSlugify(t *testing.T) {
	cases := map[string]string{"Mi Carpeta": "mi-carpeta", "  Árbol  ": "rbol", "a__b": "a-b", "": "base"}
	for in, want := range cases {
		if got := Slugify(in); got != want {
			t.Fatalf("%q -> %q, want %q", in, got, want)
		}
	}
}

func TestRejectsNestedSources(t *testing.T) {
	m := manager(t)
	parent := source(t, "vault")
	if _, err := m.Create(Meta{Name: "outer", Source: parent}); err != nil {
		t.Fatal(err)
	}
	child := filepath.Join(parent, "sub")
	if err := os.MkdirAll(child, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Create(Meta{Name: "inner", Source: child}); err == nil {
		t.Fatal("expected an overlap error")
	}
}
