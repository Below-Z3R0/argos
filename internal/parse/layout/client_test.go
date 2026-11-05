package layout

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func tempPDF(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "doc.pdf")
	if err := os.WriteFile(p, []byte("%PDF-1.4 fake"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestConvertJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/convert/file" {
			t.Errorf("path %s", r.URL.Path)
		}
		if err := r.ParseMultipartForm(8 << 20); err != nil {
			t.Fatal(err)
		}
		if _, _, err := r.FormFile("files"); err != nil {
			t.Errorf("missing file part: %v", err)
		}
		if r.FormValue("do_ocr") != "true" {
			t.Errorf("ocr not requested: %q", r.FormValue("do_ocr"))
		}
		json.NewEncoder(w).Encode(map[string]any{
			"document": map[string]string{"md_content": "# Título\n\nTexto del PDF.\n"},
		})
	}))
	defer srv.Close()

	md, err := New(srv.URL).Convert(context.Background(), tempPDF(t))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(md, "Texto del PDF") {
		t.Fatalf("markdown %q", md)
	}
}

func TestConvertMarkdownBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/markdown")
		io.WriteString(w, "## Sección\n\ncuerpo\n")
	}))
	defer srv.Close()

	md, err := New(srv.URL).Convert(context.Background(), tempPDF(t))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(md, "Sección") {
		t.Fatalf("markdown %q", md)
	}
}

func TestConvertErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	if _, err := New(srv.URL).Convert(context.Background(), tempPDF(t)); err == nil {
		t.Fatal("expected error")
	}
}

func TestConvertWithoutURL(t *testing.T) {
	if _, err := New("").Convert(context.Background(), tempPDF(t)); err == nil {
		t.Fatal("expected error")
	}
}
