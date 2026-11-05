package parse

import (
	"archive/zip"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func writeFile(t *testing.T, name, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestTextParser(t *testing.T) {
	doc, err := Text{}.Parse(context.Background(), writeFile(t, "nota.txt", "línea uno\nlínea dos\n"), "nota.txt")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Title != "nota" {
		t.Fatalf("title %q", doc.Title)
	}
	if len(doc.Sections) != 1 || !strings.Contains(doc.Sections[0].Body(), "línea dos") {
		t.Fatalf("sections %+v", doc.Sections)
	}
}

func TestHTMLHeadings(t *testing.T) {
	body := `<html><body>
	<h1>Guía</h1><p>Intro.</p>
	<h2>Instalación</h2><p>Paso uno.</p>
	<script>var x = 1;</script>
	<h3>Notas</h3><p>Ojo.</p>
	</body></html>`
	doc, err := HTML{}.Parse(context.Background(), writeFile(t, "g.html", body), "g.html")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Title != "Guía" {
		t.Fatalf("title %q", doc.Title)
	}
	if len(doc.Sections) != 3 {
		t.Fatalf("sections %d: %+v", len(doc.Sections), doc.Sections)
	}
	if got := doc.Sections[1].Path; got != "Guía > Instalación" {
		t.Fatalf("breadcrumb %q", got)
	}
	if strings.Contains(doc.Sections[1].Body(), "var x") {
		t.Fatal("script content leaked")
	}
}

func writeDOCX(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "doc.docx")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for name, content := range map[string]string{
		"word/document.xml": `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>` + body + `</w:body></w:document>`,
	} {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return path
}

func TestDOCXHeadings(t *testing.T) {
	body := `<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Manual</w:t></w:r></w:p>
	<w:p><w:r><w:t>Texto de intro.</w:t></w:r></w:p>
	<w:p><w:pPr><w:pStyle w:val="Ttulo2"/></w:pPr><w:r><w:t>Sección</w:t></w:r></w:p>
	<w:p><w:r><w:t>Contenido.</w:t></w:r></w:p>`
	doc, err := DOCX{}.Parse(context.Background(), writeDOCX(t, body), "doc.docx")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Title != "Manual" {
		t.Fatalf("title %q", doc.Title)
	}
	if len(doc.Sections) != 2 {
		t.Fatalf("sections %d: %+v", len(doc.Sections), doc.Sections)
	}
	if got := doc.Sections[1].Path; got != "Manual > Sección" {
		t.Fatalf("breadcrumb %q", got)
	}
}

func writeEPUB(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "libro.epub")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	w, err := zw.Create("OEBPS/chap1.xhtml")
	if err != nil {
		t.Fatal(err)
	}
	w.Write([]byte(`<html><body><h1>Capítulo</h1><p>Texto.</p></body></html>`))
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return path
}

func TestEPUBChapter(t *testing.T) {
	doc, err := EPUB{}.Parse(context.Background(), writeEPUB(t), "libro.epub")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Title != "Capítulo" {
		t.Fatalf("title %q", doc.Title)
	}
	if len(doc.Sections) != 1 || !strings.Contains(doc.Sections[0].Body(), "Texto") {
		t.Fatalf("sections %+v", doc.Sections)
	}
}

func TestPDFRequiresBinario(t *testing.T) {
	if _, err := exec.LookPath("pdftotext"); err == nil {
		t.Skip("pdftotext instalado: la extracción real se prueba sobre el vault")
	}
	_, err := PDF{}.Parse(context.Background(), writeFile(t, "x.pdf", "%PDF-1.4"), "x.pdf")
	if err == nil || !strings.Contains(err.Error(), "pdftotext") {
		t.Fatalf("expected a clear error, got %v", err)
	}
}

func TestRegistryKnowsEveryFormat(t *testing.T) {
	reg := DefaultRegistry()
	for _, ext := range []string{".md", ".txt", ".html", ".docx", ".epub", ".pdf"} {
		if _, ok := reg.Lookup("/tmp/file" + ext); !ok {
			t.Fatalf("%s not registered", ext)
		}
	}
}
