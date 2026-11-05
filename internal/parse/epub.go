package parse

import (
	"archive/zip"
	"context"
	"fmt"
	"sort"
	"strings"
)

// EPUB reads the XHTML chapters of an EPUB container.
//
// Chapter order comes from the file names inside the archive. It is not a full
// OPF spine reader: for note-style EPUBs that is enough, and it avoids parsing
// three more XML layers for a marginal gain.
type EPUB struct{}

// Extensions implements Parser.
func (EPUB) Extensions() []string { return []string{".epub"} }

// Parse implements Parser.
func (EPUB) Parse(ctx context.Context, path, rel string) (Document, error) {
	if err := ctx.Err(); err != nil {
		return Document{}, err
	}
	zr, err := zip.OpenReader(path)
	if err != nil {
		return Document{}, err
	}
	defer zr.Close()

	var names []string
	for _, f := range zr.File {
		lower := strings.ToLower(f.Name)
		if strings.HasSuffix(lower, ".xhtml") || strings.HasSuffix(lower, ".html") || strings.HasSuffix(lower, ".htm") {
			names = append(names, f.Name)
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		return Document{}, fmt.Errorf("epub %s: no XHTML chapters", rel)
	}

	var sections []Section
	for _, name := range names {
		f, err := zr.Open(name)
		if err != nil {
			return Document{}, err
		}
		chapter, err := sectionsFromHTML(f)
		f.Close()
		if err != nil {
			return Document{}, fmt.Errorf("epub %s: %s: %w", rel, name, err)
		}
		sections = append(sections, chapter...)
	}

	title := strings.TrimSuffix(baseName(rel), extOf(rel))
	if len(sections) > 0 && sections[0].Heading != "" {
		title = sections[0].Heading
	}
	return Document{Path: path, Rel: rel, Title: title, Meta: map[string]string{}, Sections: sections}, nil
}
