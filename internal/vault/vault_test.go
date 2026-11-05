package vault

import (
	"os"
	"path/filepath"
	"testing"
)

func tree(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := []string{
		"a.md", "b.md", "c.txt", "sub/d.md", "sub/e.pdf",
		".git/config.md", "node_modules/x.md", ".obsidian/y.md",
	}
	for _, f := range files {
		p := filepath.Join(root, f)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func names(files []File) []string {
	out := make([]string, len(files))
	for i, f := range files {
		out[i] = f.Rel
	}
	return out
}

func TestDefaultOnlyMarkdownAndSkipsWorkDirs(t *testing.T) {
	files, err := Walk(tree(t), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	got := names(files)
	want := []string{"a.md", "b.md", "sub/d.md"}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}

func TestIncludePDFAndMD(t *testing.T) {
	opts := DefaultOptions()
	opts.Include = []string{"md", ".pdf"}
	files, err := Walk(tree(t), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 4 {
		t.Fatalf("got %v", names(files))
	}
}

func TestExcludeWins(t *testing.T) {
	opts := DefaultOptions()
	opts.Include = []string{".md", ".txt"}
	opts.Exclude = []string{"txt"}
	files, _ := Walk(tree(t), opts)
	for _, f := range files {
		if f.Ext == ".txt" {
			t.Fatalf("excluded extension returned: %v", names(files))
		}
	}
}

func TestEmptyIncludeMeansAll(t *testing.T) {
	opts := DefaultOptions()
	opts.Include = nil
	files, _ := Walk(tree(t), opts)
	if len(files) != 5 { // a.md b.md c.txt sub/d.md sub/e.pdf
		t.Fatalf("got %v", names(files))
	}
}

func TestSummarize(t *testing.T) {
	files := []File{{Ext: ".md", Size: 10}, {Ext: ".md", Size: 20}, {Ext: ".pdf", Size: 5}}
	s := Summarize(files)
	if s.Total != 3 || s.Bytes != 35 {
		t.Fatalf("total %d bytes %d", s.Total, s.Bytes)
	}
	if s.ByExt[".md"].Files != 2 || s.ByExt[".md"].Bytes != 30 {
		t.Fatalf("md stat %+v", s.ByExt[".md"])
	}
}

func TestNormalizeExt(t *testing.T) {
	cases := map[string]string{"md": ".md", ".MD": ".md", "": "", " .Pdf ": ".pdf"}
	for in, want := range cases {
		if got := NormalizeExt(in); got != want {
			t.Fatalf("%q -> %q, want %q", in, got, want)
		}
	}
}
