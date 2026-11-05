package rerank

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestScoresRelevantHigher runs the real model when ARGOS_RERANKER_DIR points
// to the installed bge reranker. It is the cheapest way to catch a broken pair
// template: with a wrong [CLS]/[SEP] layout the model still runs, but the
// ordering stops making sense.
func TestScoresRelevantHigher(t *testing.T) {
	dir := os.Getenv("ARGOS_RERANKER_DIR")
	if dir == "" {
		t.Skip("set ARGOS_RERANKER_DIR to run the model test")
	}
	r, err := New(Options{
		ModelPath:     filepath.Join(dir, "model.onnx"),
		TokenizerPath: filepath.Join(dir, "tokenizer.json"),
		LibraryPath:   os.Getenv("ARGOS_ORT_LIB"),
		Device:        "cpu",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	query := "¿qué es un gato?"
	passages := []string{
		"Un gato es un animal doméstico de la familia de los félidos.",
		"El índice bursátil subió tres puntos porcentuales hoy.",
	}
	scores, err := r.Score(context.Background(), query, passages)
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 2 {
		t.Fatalf("scores %v", scores)
	}
	for _, s := range scores {
		if s < 0 || s > 1 {
			t.Fatalf("score out of range: %v", scores)
		}
	}
	if scores[0] <= scores[1] {
		t.Fatalf("relevant passage should score higher: %v", scores)
	}
}
