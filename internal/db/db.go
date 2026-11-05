// Package db manages the independent bases: each one has a source folder, a
// collection, a model and its own manifest.
package db

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Meta is the identity of a base. It never changes after creation.
type Meta struct {
	Name         string    `json:"name"`
	Source       string    `json:"source"`
	Collection   string    `json:"collection"`
	Model        string    `json:"model"`
	Dims         int       `json:"dims"`
	Quantization string    `json:"quantization"`
	Created      time.Time `json:"created"`
}

// Manager owns the dbs directory.
type Manager struct {
	Root string
}

// New returns a manager rooted at root.
func New(root string) *Manager { return &Manager{Root: root} }

// Dir is where a base lives.
func (m *Manager) Dir(name string) string { return filepath.Join(m.Root, name) }

// ManifestPath is the bbolt file of a base.
func (m *Manager) ManifestPath(name string) string {
	return filepath.Join(m.Dir(name), "manifest.db")
}

// Get reads one base.
func (m *Manager) Get(name string) (Meta, error) {
	data, err := os.ReadFile(filepath.Join(m.Dir(name), "meta.json"))
	if err != nil {
		return Meta{}, fmt.Errorf("db %s: %w", name, err)
	}
	var meta Meta
	if err := json.Unmarshal(data, &meta); err != nil {
		return Meta{}, fmt.Errorf("db %s: %w", name, err)
	}
	return meta, nil
}

// List returns every base, sorted by name.
func (m *Manager) List() ([]Meta, error) {
	entries, err := os.ReadDir(m.Root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []Meta
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		meta, err := m.Get(e.Name())
		if err != nil {
			continue // an incomplete directory is not a base
		}
		out = append(out, meta)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// Create registers a base. If the name is taken, a short random suffix is
// appended and the final name is returned.
func (m *Manager) Create(meta Meta) (Meta, error) {
	if meta.Source == "" {
		return Meta{}, fmt.Errorf("db: source is required")
	}
	if abs, err := filepath.Abs(meta.Source); err == nil {
		meta.Source = abs
	}
	if st, err := os.Stat(meta.Source); err != nil || !st.IsDir() {
		return Meta{}, fmt.Errorf("db: source %s is not a directory", meta.Source)
	}
	// Two bases must not cover the same tree: indexing a folder twice wastes
	// embeddings and makes search return duplicates.
	existing, _ := m.List()
	for _, other := range existing {
		if within(meta.Source, other.Source) || within(other.Source, meta.Source) {
			return Meta{}, fmt.Errorf("db: %s se superpone con la base %s (%s)", meta.Source, other.Name, other.Source)
		}
	}
	if meta.Name == "" {
		meta.Name = Slugify(filepath.Base(meta.Source))
	}
	meta.Name = m.unique(meta.Name)
	if meta.Collection == "" {
		meta.Collection = "argos_" + meta.Name
	}
	if meta.Created.IsZero() {
		meta.Created = time.Now().UTC()
	}
	dir := m.Dir(meta.Name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Meta{}, err
	}
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return Meta{}, err
	}
	if err := os.WriteFile(filepath.Join(dir, "meta.json"), data, 0o644); err != nil {
		return Meta{}, err
	}
	return meta, nil
}

// Remove deletes the metadata directory. The collection is the caller's job:
// this package does not talk to Qdrant.
func (m *Manager) Remove(name string) error {
	if _, err := m.Get(name); err != nil {
		return err
	}
	return os.RemoveAll(m.Dir(name))
}

// within reports whether child is parent or lives under it.
func within(child, parent string) bool {
	if child == parent {
		return true
	}
	return strings.HasPrefix(child, parent+string(filepath.Separator))
}

// unique appends a 4-char suffix until the name is free.
func (m *Manager) unique(name string) string {
	if _, err := os.Stat(m.Dir(name)); os.IsNotExist(err) {
		return name
	}
	for i := 0; i < 20; i++ {
		buf := make([]byte, 2)
		_, _ = rand.Read(buf)
		candidate := name + "-" + hex.EncodeToString(buf)
		if _, err := os.Stat(m.Dir(candidate)); os.IsNotExist(err) {
			return candidate
		}
	}
	return fmt.Sprintf("%s-%d", name, time.Now().UnixNano())
}

// Slugify turns a folder name into a usable base name.
func Slugify(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastDash = false
		case r == '-' || r == '_' || r == ' ' || r == '.':
			if !lastDash && b.Len() > 0 {
				b.WriteByte('-')
				lastDash = true
			}
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "base"
	}
	return out
}
