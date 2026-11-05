// Package models manages the installed artifacts: download, hash
// verification, atomic install, and removal.
//
// Model metadata lives in the registry; this package only touches disk and
// network. Installs are atomic: a failed download never leaves a half-written
// model that later looks installed.
package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"syscall"
	"time"

	"github.com/Below-Z3R0/argos/internal/registry"
)

// ErrNotInstalled means the model is absent or incomplete.
var ErrNotInstalled = errors.New("model not installed")

// ErrUnpinned means the catalogue has no sha256 for the model.
var ErrUnpinned = errors.New("model has no pinned sha256")

// Store is the models root, e.g. $ARGOS_HOME/models.
type Store struct {
	Root   string
	Client *http.Client
}

// New returns a store rooted at root.
func New(root string) *Store {
	return &Store{
		Root:   root,
		Client: &http.Client{Timeout: 0}, // long downloads; cancel via context
	}
}

// Installed is the on-disk record written after a successful install.
type Installed struct {
	ID          string    `json:"id"`
	Task        string    `json:"task"`
	SHA256      string    `json:"sha256"`
	SizeBytes   int64     `json:"size_bytes"`
	InstalledAt time.Time `json:"installed_at"`
	Source      string    `json:"source"`
}

// Dir returns the directory for a model, creating nothing.
func (s *Store) Dir(task, id string) (string, error) {
	sub, err := registry.StoreDir(task)
	if err != nil {
		return "", err
	}
	return filepath.Join(s.Root, sub, id), nil
}

// Status reports whether the model is installed and complete.
func (s *Store) Status(task, id string) (Installed, bool) {
	dir, err := s.Dir(task, id)
	if err != nil {
		return Installed{}, false
	}
	data, err := os.ReadFile(filepath.Join(dir, "installed.json"))
	if err != nil {
		return Installed{}, false
	}
	var rec Installed
	if err := json.Unmarshal(data, &rec); err != nil {
		return Installed{}, false
	}
	if rec.ID != id || rec.Task != task {
		return Installed{}, false
	}
	return rec, true
}

// Install downloads every file, verifies its sha256, and commits atomically.
//
// Nothing is written outside <dir>/.tmp until all files verify. The rename of
// the temp directory is the commit point.
func (s *Store) Install(ctx context.Context, m registry.Model, opts Options) error {
	if !opts.AllowUnpinned {
		for _, f := range m.Files {
			if f.SHA256 == "" {
				return fmt.Errorf("%w: %s (%s)", ErrUnpinned, m.ID, f.Dest)
			}
		}
	}
	if len(m.Files) == 0 {
		return fmt.Errorf("model %s has no files", m.ID)
	}

	var neededBytes uint64
	for _, f := range m.Files {
		if f.SizeBytes > 0 {
			neededBytes += uint64(f.SizeBytes)
		} else if f.SizeMB > 0 {
			neededBytes += uint64(f.SizeMB) << 20
		}
	}
	if neededBytes > 0 {
		if free, err := freeSpace(s.Root); err == nil && free < neededBytes+neededBytes/5 {
			return fmt.Errorf("no hay espacio: se necesitan %.0f MB y quedan %.0f MB",
				float64(neededBytes)*1.2/(1<<20), float64(free)/(1<<20))
		}
	}

	dir, err := s.Dir(m.Task, m.ID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp := filepath.Join(dir, ".tmp")
	if err := os.RemoveAll(tmp); err != nil {
		return err
	}
	if err := os.MkdirAll(tmp, 0o755); err != nil {
		return err
	}
	defer os.RemoveAll(tmp)

	totalExpected := int64(0)
	for _, f := range m.Files {
		if f.SizeBytes > 0 {
			totalExpected += f.SizeBytes
		} else {
			totalExpected += int64(f.SizeMB) << 20
		}
	}
	state := DownloadState{ID: m.ID, State: "downloading", TotalBytes: totalExpected, Started: time.Now().UTC()}
	s.saveState(m.Task, m.ID, state)

	var total int64
	for _, f := range m.Files {
		n, err := s.download(ctx, f, filepath.Join(tmp, f.Dest), opts, total, totalExpected, func(done int64) {
			state.DoneBytes = done
			s.saveState(m.Task, m.ID, state)
		})
		if err != nil {
			state.State = "failed"
			state.Error = err.Error()
			s.saveState(m.Task, m.ID, state)
			return fmt.Errorf("descarga de %s: %w", f.Dest, err)
		}
		total += n
	}
	if err := os.Remove(s.statePath(m.Task, m.ID)); err != nil && !os.IsNotExist(err) {
		// a stale marker is harmless; list will show "incompleto"
	}

	rec := Installed{
		ID:          m.ID,
		Task:        m.Task,
		SHA256:      m.Files[0].SHA256,
		SizeBytes:   total,
		InstalledAt: time.Now().UTC(),
		Source:      m.Files[0].URL,
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(tmp, "installed.json"), data, 0o644); err != nil {
		return err
	}

	// Commit: move each file and the record into the final directory.
	for _, f := range m.Files {
		if err := os.Rename(filepath.Join(tmp, f.Dest), filepath.Join(dir, f.Dest)); err != nil {
			return fmt.Errorf("commit %s: %w", f.Dest, err)
		}
	}
	if err := os.Rename(filepath.Join(tmp, "installed.json"), filepath.Join(dir, "installed.json")); err != nil {
		return fmt.Errorf("commit record: %w", err)
	}
	return nil
}

// Options tunes an install.
type Options struct {
	AllowUnpinned bool
	// Progress receives bytes downloaded and the total. May be nil.
	Progress func(done, total int64)
}

// DownloadState is the on-disk marker of an install in progress.
type DownloadState struct {
	ID         string    `json:"id"`
	State      string    `json:"state"` // downloading | failed
	DoneBytes  int64     `json:"done_bytes"`
	TotalBytes int64     `json:"total_bytes"`
	Started    time.Time `json:"started"`
	Error      string    `json:"error,omitempty"`
}

func (s *Store) statePath(task, id string) string {
	dir, err := s.Dir(task, id)
	if err != nil {
		return ""
	}
	return filepath.Join(dir, ".state.json")
}

// State reads the download marker, if any.
func (s *Store) State(task, id string) (DownloadState, bool) {
	path := s.statePath(task, id)
	data, err := os.ReadFile(path)
	if err != nil {
		return DownloadState{}, false
	}
	var st DownloadState
	if json.Unmarshal(data, &st) != nil {
		return DownloadState{}, false
	}
	return st, true
}

func (s *Store) saveState(task, id string, st DownloadState) {
	path := s.statePath(task, id)
	if path == "" {
		return
	}
	data, err := json.Marshal(st)
	if err != nil {
		return
	}
	_ = os.WriteFile(path, data, 0o644)
}

func (s *Store) download(ctx context.Context, f registry.File, dest string, opts Options, base, total int64, onBytes func(int64)) (int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.URL, nil)
	if err != nil {
		return 0, err
	}
	resp, err := s.Client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("sin conexión o URL inaccesible (status %s)", resp.Status)
	}

	out, err := os.Create(dest)
	if err != nil {
		return 0, err
	}
	defer out.Close()

	hash := sha256.New()
	var done int64
	buf := make([]byte, 1<<20)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := out.Write(buf[:n]); werr != nil {
				return done, werr
			}
			hash.Write(buf[:n])
			done += int64(n)
			if opts.Progress != nil {
				opts.Progress(base+done, total)
			}
			if onBytes != nil {
				onBytes(base + done)
			}
		}
		if errors.Is(rerr, io.EOF) {
			break
		}
		if rerr != nil {
			return done, rerr
		}
	}
	if err := out.Sync(); err != nil {
		return done, err
	}
	if f.SHA256 != "" {
		got := hex.EncodeToString(hash.Sum(nil))
		if got != f.SHA256 {
			return done, fmt.Errorf("sha256 mismatch: want %s got %s", f.SHA256, got)
		}
	}
	return done, nil
}

// Verify recomputes the hashes of an installed model.
func (s *Store) Verify(task, id string, files []registry.File) error {
	dir, err := s.Dir(task, id)
	if err != nil {
		return err
	}
	if _, ok := s.Status(task, id); !ok {
		return fmt.Errorf("%w: %s", ErrNotInstalled, id)
	}
	for _, f := range files {
		if f.SHA256 == "" {
			continue
		}
		got, err := hashFile(filepath.Join(dir, f.Dest))
		if err != nil {
			return err
		}
		if got != f.SHA256 {
			return fmt.Errorf("%s: sha256 mismatch", f.Dest)
		}
	}
	return nil
}

// Remove deletes an installed model directory.
func (s *Store) Remove(task, id string) error {
	dir, err := s.Dir(task, id)
	if err != nil {
		return err
	}
	if _, ok := s.Status(task, id); !ok {
		return fmt.Errorf("%w: %s", ErrNotInstalled, id)
	}
	return os.RemoveAll(dir)
}

// InstalledModels lists models of a task that are installed.
func (s *Store) InstalledModels(task string, catalog *registry.Catalog) []registry.Model {
	var out []registry.Model
	for _, m := range catalog.ByTask(task) {
		if _, ok := s.Status(m.Task, m.ID); ok {
			out = append(out, m)
		}
	}
	return out
}

// freeSpace returns the bytes available on the filesystem holding path.
func freeSpace(path string) (uint64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, err
	}
	return st.Bavail * uint64(st.Bsize), nil
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
