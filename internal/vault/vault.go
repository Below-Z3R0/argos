// Package vault walks the corpus and decides which files belong to an index.
//
// Filters live here, not in the indexer: the same walk feeds dry-runs, counts
// and the future incremental diff.
package vault

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Options decide what is included.
type Options struct {
	// Include is the set of extensions to keep (".md", "pdf"). Empty means all.
	Include []string
	// Exclude removes specific extensions from the included set.
	Exclude []string
	// ExcludeDirs skips directories by base name at any depth.
	ExcludeDirs []string
	// MaxSize skips files larger than this many bytes (0 = no limit).
	MaxSize int64
}

// DefaultOptions is the project default: only Markdown.
func DefaultOptions() Options {
	return Options{
		Include:     []string{".md"},
		ExcludeDirs: []string{".git", ".obsidian", ".rag", ".trash", "node_modules", "venv", ".venv"},
	}
}

// File is one candidate for indexing.
type File struct {
	Path  string // absolute
	Rel   string // relative to the root, slash separated
	Ext   string // lowercase, with dot
	Size  int64
	MTime int64 // unix seconds
}

// NormalizeExt accepts "md", ".MD" or ".md" and returns ".md".
func NormalizeExt(ext string) string {
	ext = strings.ToLower(strings.TrimSpace(ext))
	if ext == "" {
		return ""
	}
	if !strings.HasPrefix(ext, ".") {
		ext = "." + ext
	}
	return ext
}

func normalizeList(values []string) map[string]bool {
	out := map[string]bool{}
	for _, v := range values {
		if v = NormalizeExt(v); v != "" {
			out[v] = true
		}
	}
	return out
}

func dirSet(values []string) map[string]bool {
	out := map[string]bool{}
	for _, v := range values {
		if v = strings.TrimSpace(v); v != "" {
			out[v] = true
		}
	}
	return out
}

// Walk returns the files under root that pass the filters, sorted by path.
func Walk(root string, opts Options) ([]File, error) {
	st, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("vault: %w", err)
	}
	if !st.IsDir() {
		return nil, errors.New("vault: root is not a directory")
	}
	include := normalizeList(opts.Include)
	exclude := normalizeList(opts.Exclude)
	skipDirs := dirSet(opts.ExcludeDirs)

	var files []File
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// A permission error on one subtree must not abort the walk.
			if errors.Is(err, fs.ErrPermission) {
				if d != nil && d.IsDir() {
					return fs.SkipDir
				}
				return nil
			}
			return err
		}
		if d.IsDir() {
			if path != root && skipDirs[d.Name()] {
				return fs.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		ext := NormalizeExt(filepath.Ext(d.Name()))
		if len(include) > 0 && !include[ext] {
			return nil
		}
		if exclude[ext] {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		if opts.MaxSize > 0 && info.Size() > opts.MaxSize {
			return nil
		}
		rel, rerr := filepath.Rel(root, path)
		if rerr != nil {
			return nil
		}
		files = append(files, File{
			Path:  path,
			Rel:   filepath.ToSlash(rel),
			Ext:   ext,
			Size:  info.Size(),
			MTime: info.ModTime().Unix(),
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Rel < files[j].Rel })
	return files, nil
}

// Summary aggregates a walk by extension.
type Summary struct {
	Total int
	Bytes int64
	ByExt map[string]ExtStat
}

// ExtStat is the count and size of one extension.
type ExtStat struct {
	Files int
	Bytes int64
}

// Summarize groups files by extension.
func Summarize(files []File) Summary {
	s := Summary{ByExt: map[string]ExtStat{}}
	for _, f := range files {
		e := s.ByExt[f.Ext]
		e.Files++
		e.Bytes += f.Size
		s.ByExt[f.Ext] = e
		s.Total++
		s.Bytes += f.Size
	}
	return s
}

// Hash returns the sha256 of a file's content.
func Hash(path string) (string, error) {
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
