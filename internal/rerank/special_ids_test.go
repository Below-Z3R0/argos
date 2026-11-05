package rerank

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTokenizer(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestSpecialIDsUnigram(t *testing.T) {
	path := writeTokenizer(t, `{"model":{"vocab":[["<unk>",0],["<s>",0],["</s>",0],["hello",0]]}}`)
	ids, err := specialIDs(path)
	if err != nil {
		t.Fatal(err)
	}
	if ids[0] != 1 || ids[1] != 2 {
		t.Fatalf("ids = %v, want [1 2]", ids)
	}
}

func TestSpecialIDsBPE(t *testing.T) {
	path := writeTokenizer(t, `{"model":{"vocab":{"<s>":5,"</s>":6,"hello":7}}}`)
	ids, err := specialIDs(path)
	if err != nil {
		t.Fatal(err)
	}
	if ids[0] != 5 || ids[1] != 6 {
		t.Fatalf("ids = %v, want [5 6]", ids)
	}
}

func TestSpecialIDsAddedTokens(t *testing.T) {
	path := writeTokenizer(t, `{"added_tokens":[{"id":9,"content":"<s>"},{"id":10,"content":"</s>"}],"model":{"vocab":{"x":1}}}`)
	ids, err := specialIDs(path)
	if err != nil {
		t.Fatal(err)
	}
	if ids[0] != 9 || ids[1] != 10 {
		t.Fatalf("ids = %v, want [9 10]", ids)
	}
}

func TestSpecialIDsMissingMarkers(t *testing.T) {
	path := writeTokenizer(t, `{"model":{"vocab":{"x":1}}}`)
	if _, err := specialIDs(path); err == nil {
		t.Fatal("expected an error for a tokenizer without <s> and </s>")
	}
}
