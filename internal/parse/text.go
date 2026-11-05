package parse

import (
	"context"
	"os"
	"strings"
)

// Text handles plain text files: one section, no structure to recover.
type Text struct{}

// Extensions implements Parser.
func (Text) Extensions() []string { return []string{".txt", ".text"} }

// Parse reads the file as a single section.
func (Text) Parse(ctx context.Context, path, rel string) (Document, error) {
	if err := ctx.Err(); err != nil {
		return Document{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Document{}, err
	}
	title := strings.TrimSuffix(baseName(rel), extOf(rel))
	return Document{
		Path:  path,
		Rel:   rel,
		Title: title,
		Meta:  map[string]string{},
		Sections: []Section{{
			Level:   0,
			Heading: title,
			Path:    title,
			Lines:   strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n"),
		}},
	}, nil
}
