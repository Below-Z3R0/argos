package parse

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "nota.md")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

const sample = `---
title: Notas de Docker
lang: es
tags: [docker, redes]
---
Texto previo al primer título.

# Notas de Docker

Introducción.

## Redes

Las redes bridge aíslan.

### Puertos

Se publican con -p.
`

func TestMarkdownFrontmatterAndSections(t *testing.T) {
	doc, err := Markdown{}.Parse(context.Background(), write(t, sample), "notas/docker.md")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Title != "Notas de Docker" || doc.Lang != "es" {
		t.Fatalf("meta: %+v", doc)
	}
	if len(doc.Tags) != 2 || doc.Tags[0] != "docker" {
		t.Fatalf("tags: %v", doc.Tags)
	}
	if len(doc.Sections) != 4 {
		t.Fatalf("sections: %d", len(doc.Sections))
	}
	if got := doc.Sections[0]; got.Level != 0 || !strings.Contains(got.Body(), "Texto previo") {
		t.Fatalf("preamble lost: %+v", got)
	}
	if got := doc.Sections[2].Path; got != "Notas de Docker > Redes" {
		t.Fatalf("breadcrumb: %q", got)
	}
	if got := doc.Sections[3].Path; got != "Notas de Docker > Redes > Puertos" {
		t.Fatalf("nested breadcrumb: %q", got)
	}
	if strings.Contains(doc.Sections[1].Body(), "# Notas") {
		t.Fatal("heading line leaked into the body")
	}
}

func TestMarkdownTitleFallsBackToHeading(t *testing.T) {
	doc, err := Markdown{}.Parse(context.Background(), write(t, "# Solo un título\n\ncuerpo\n"), "x.md")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Title != "Solo un título" {
		t.Fatalf("title %q", doc.Title)
	}
}

func TestMarkdownTitleFallsBackToFilename(t *testing.T) {
	doc, err := Markdown{}.Parse(context.Background(), write(t, "sin títulos\n"), "carpeta/mi-nota.md")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Title != "mi-nota" {
		t.Fatalf("title %q", doc.Title)
	}
}

func TestUnclosedFrontmatterIsBody(t *testing.T) {
	doc, err := Markdown{}.Parse(context.Background(), write(t, "---\ntitle: x\nsin cierre\n"), "x.md")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Title != "x" {
		// title comes from the filename when the block is not closed
		t.Fatalf("title %q", doc.Title)
	}
}

func TestHeadingNeedsSpace(t *testing.T) {
	doc, _ := Markdown{}.Parse(context.Background(), write(t, "#NoEsHeading\n\ntexto\n"), "x.md")
	if len(doc.Sections) != 1 {
		t.Fatalf("expected one section, got %d", len(doc.Sections))
	}
}

func TestRegistryDispatch(t *testing.T) {
	reg := DefaultRegistry()
	if _, ok := reg.Lookup("/tmp/a.md"); !ok {
		t.Fatal("md should be registered")
	}
	if _, ok := reg.Lookup("/tmp/a.pdf"); !ok {
		t.Fatal("pdf should be registered")
	}
	_, err := reg.Parse(context.Background(), "/tmp/a.xyz", "a.xyz")
	if !errors.Is(err, ErrNoParser) {
		t.Fatalf("expected ErrNoParser, got %v", err)
	}
	if exts := reg.Extensions(); len(exts) < 6 {
		t.Fatalf("extensions: %v", exts)
	}
}

func TestHeadingInsideCodeFence(t *testing.T) {
	body := "# Real\n\ntexto\n\n```bash\n# esto es un comentario de shell\n## tampoco es heading\n```\n\n## Sección\n\nmás texto\n"
	doc, err := Markdown{}.Parse(context.Background(), write(t, body), "x.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, sec := range doc.Sections {
		if strings.Contains(sec.Heading, "esto es un comentario") || strings.Contains(sec.Heading, "tampoco es heading") {
			t.Fatalf("fence content parsed as heading: %+v", sec)
		}
	}
	var found bool
	for _, sec := range doc.Sections {
		if sec.Heading == "Sección" {
			found = true
		}
	}
	if !found {
		t.Fatalf("real heading lost: %+v", doc.Sections)
	}
}
