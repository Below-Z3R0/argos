// Package manifest records what has been indexed: which file produced which
// chunks. It is the control plane for incremental indexing, not a copy of the
// vectors.
package manifest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	bolt "go.etcd.io/bbolt"
)

var bucket = []byte("files")

// ChunkRef is the identity and content hash of one indexed chunk.
type ChunkRef struct {
	ID   string `json:"id"`
	Hash string `json:"hash"`
}

// FileState is what one file left in the index.
type FileState struct {
	Rel     string     `json:"rel"`
	Hash    string     `json:"hash"`
	MTime   int64      `json:"mtime"`
	Size    int64      `json:"size"`
	Model   string     `json:"model"`
	Chunks  []ChunkRef `json:"chunks"`
	Indexed time.Time  `json:"indexed_at"`
}

// Manifest is the bbolt-backed store of file states.
type Manifest struct {
	db *bolt.DB
}

// Open opens (or creates) the manifest at path.
func Open(path string) (*Manifest, error) {
	// The manifest lives under data/<collection>/, which may not exist yet.
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("manifest: create directory: %w", err)
	}
	db, err := bolt.Open(path, 0o644, &bolt.Options{Timeout: 5 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("manifest: open %s: %w", path, err)
	}
	err = db.Update(func(tx *bolt.Tx) error {
		_, e := tx.CreateBucketIfNotExists(bucket)
		return e
	})
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("manifest: bucket: %w", err)
	}
	return &Manifest{db: db}, nil
}

// Close closes the database.
func (m *Manifest) Close() error { return m.db.Close() }

// Get returns the state of a file.
func (m *Manifest) Get(rel string) (FileState, bool, error) {
	var st FileState
	found := false
	err := m.db.View(func(tx *bolt.Tx) error {
		raw := tx.Bucket(bucket).Get([]byte(rel))
		if raw == nil {
			return nil
		}
		found = true
		return json.Unmarshal(raw, &st)
	})
	if err != nil {
		return FileState{}, false, fmt.Errorf("manifest: get %s: %w", rel, err)
	}
	return st, found, nil
}

// Put stores the state of a file.
func (m *Manifest) Put(st FileState) error {
	raw, err := json.Marshal(st)
	if err != nil {
		return err
	}
	return m.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(bucket).Put([]byte(st.Rel), raw)
	})
}

// Delete removes a file from the manifest.
func (m *Manifest) Delete(rel string) error {
	return m.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(bucket).Delete([]byte(rel))
	})
}

// All returns every recorded file, used to detect removals.
func (m *Manifest) All() ([]FileState, error) {
	var out []FileState
	err := m.db.View(func(tx *bolt.Tx) error {
		return tx.Bucket(bucket).ForEach(func(_, v []byte) error {
			var st FileState
			if err := json.Unmarshal(v, &st); err != nil {
				return err
			}
			out = append(out, st)
			return nil
		})
	})
	if err != nil {
		return nil, fmt.Errorf("manifest: all: %w", err)
	}
	return out, nil
}
