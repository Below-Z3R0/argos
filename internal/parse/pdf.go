package parse

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
)

// PDF extracts text with poppler's pdftotext, without -layout.
//
// The spike compared both modes on ten vault PDFs: plain output had 3–15 %
// whitespace against 19–35 % with -layout, and ~10 % fewer tokens for the same
// text. PDFium was discarded: no Go backend module is published and it does not
// order the text itself.
//
// A PDF has no heading structure to recover, so it becomes one section. Complex
// layouts are routed to Docling by the indexer, not here.
type PDF struct{}

// Extensions implements Parser.
func (PDF) Extensions() []string { return []string{".pdf"} }

// Parse implements Parser.
func (PDF) Parse(ctx context.Context, path, rel string) (Document, error) {
	if _, err := exec.LookPath("pdftotext"); err != nil {
		return Document{}, errors.New("pdf: pdftotext (poppler) is not installed")
	}
	out, err := exec.CommandContext(ctx, "pdftotext", "-q", path, "-").Output()
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			return Document{}, fmt.Errorf("pdf %s: pdftotext failed: %s", rel, strings.TrimSpace(string(exit.Stderr)))
		}
		return Document{}, fmt.Errorf("pdf %s: %w", rel, err)
	}
	text := strings.ReplaceAll(string(out), "\r\n", "\n")
	text = strings.ReplaceAll(text, "\f", "\n")
	text = strings.TrimSpace(text)

	title := strings.TrimSuffix(baseName(rel), extOf(rel))
	doc := Document{Path: path, Rel: rel, Title: title, Meta: map[string]string{}}
	if text == "" {
		// No text layer: the indexer marks it as needing OCR instead of
		// indexing an empty chunk.
		doc.Meta["needs_ocr"] = "true"
		return doc, nil
	}
	doc.Sections = []Section{{
		Level:   0,
		Heading: title,
		Path:    title,
		Lines:   strings.Split(text, "\n"),
	}}
	return doc, nil
}
