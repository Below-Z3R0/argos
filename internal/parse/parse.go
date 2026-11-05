// Package parse turns files into structured documents.
//
// One parser per format, registered by extension. The indexer only knows the
// interface, so adding PDF or DOCX later is a new file plus a registration,
// not a change in the pipeline.
package parse

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// Document is the normalized output of any parser.
type Document struct {
	Path     string // absolute path on disk
	Rel      string // path relative to the vault, slash separated
	Title    string
	Lang     string
	Tags     []string
	Meta     map[string]string
	Sections []Section
}

// Section is a heading and everything under it, without the heading line.
type Section struct {
	Level   int      // 1..6; 0 for the preamble before the first heading
	Heading string   // heading text
	Path    string   // breadcrumb: "Título > Sección > Subsección"
	Lines   []string // body lines, heading excluded
}

// Body joins the section lines.
func (s Section) Body() string { return strings.Join(s.Lines, "\n") }

// Parser reads one or more extensions.
type Parser interface {
	Extensions() []string
	Parse(ctx context.Context, path, rel string) (Document, error)
}

// ErrNoParser means no registered parser handles the extension.
var ErrNoParser = errors.New("no parser for extension")

// Registry maps extensions to parsers.
type Registry struct {
	byExt map[string]Parser
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return &Registry{byExt: map[string]Parser{}} }

// Register adds a parser for each of its extensions.
func (r *Registry) Register(p Parser) {
	for _, ext := range p.Extensions() {
		r.byExt[NormalizeExt(ext)] = p
	}
}

// Lookup returns the parser for a path, if any.
func (r *Registry) Lookup(path string) (Parser, bool) {
	p, ok := r.byExt[NormalizeExt(filepath.Ext(path))]
	return p, ok
}

// Parse dispatches by extension.
func (r *Registry) Parse(ctx context.Context, path, rel string) (Document, error) {
	p, ok := r.Lookup(path)
	if !ok {
		return Document{}, fmt.Errorf("%w: %s", ErrNoParser, filepath.Ext(path))
	}
	return p.Parse(ctx, path, rel)
}

// Extensions lists the registered extensions, sorted.
func (r *Registry) Extensions() []string {
	out := make([]string, 0, len(r.byExt))
	for ext := range r.byExt {
		out = append(out, ext)
	}
	sort.Strings(out)
	return out
}

// DefaultRegistry has every parser that ships today.
func DefaultRegistry() *Registry {
	r := NewRegistry()
	r.Register(Markdown{})
	r.Register(Text{})
	r.Register(HTML{})
	r.Register(DOCX{})
	r.Register(EPUB{})
	r.Register(PDF{})
	return r
}

// NormalizeExt accepts "md", ".MD" and returns ".md".
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
