package embed

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestGoldenVectors is the spike's regression test: it runs the real model and
// compares against vectors produced by the Python reference on the same
// onnxruntime version. A change in tokenization or pooling moves these numbers
// and fails here, instead of silently degrading every search.
//
// Set ARGOS_MODEL_DIR to the installed qwen3-embed-0.6b directory to run it.
func TestGoldenVectors(t *testing.T) {
	dir := os.Getenv("ARGOS_MODEL_DIR")
	if dir == "" {
		t.Skip("set ARGOS_MODEL_DIR to run the model test")
	}
	lib := os.Getenv("ARGOS_ORT_LIB")

	raw, err := os.ReadFile(filepath.Join("testdata", "golden_vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Texts   []string    `json:"texts"`
		Vectors [][]float32 `json:"vectors"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}

	e, err := NewORT(ORTOptions{
		ModelPath:     filepath.Join(dir, "model.onnx"),
		TokenizerPath: filepath.Join(dir, "tokenizer.json"),
		LibraryPath:   lib,
		Name:          "qwen3-embed-0.6b",
		Dims:          1024,
		MaxTokens:     512,
		Pooling:       "last_token",
	})
	if err != nil {
		t.Fatalf("open model: %v", err)
	}
	defer e.Close()

	got, err := e.EmbedPassages(context.Background(), golden.Texts)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(got) != len(golden.Vectors) {
		t.Fatalf("got %d vectors, want %d", len(got), len(golden.Vectors))
	}
	for i := range got {
		c := cosine(got[i], golden.Vectors[i])
		if c < 0.9999 {
			t.Errorf("text %d: cosine %.6f with the reference, want >= 0.9999", i, c)
		}
		if math.Abs(norm(got[i])-1) > 1e-4 {
			t.Errorf("text %d: vector is not normalized (norm %.6f)", i, norm(got[i]))
		}
	}
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

func TestPoolTruncatesMatryoshka(t *testing.T) {
	o := &ORT{modelDims: 4, dims: 2, opts: ORTOptions{Pooling: "cls"}}
	data := []float32{3, 4, 100, 100, 6, 8, 0, 0}
	vec := o.pool(data, 2)
	if len(vec) != 2 {
		t.Fatalf("len %d, want 2", len(vec))
	}
	// 3,4 normalizes to 0.6,0.8; the discarded tail must not leak in.
	if math.Abs(float64(vec[0])-0.6) > 1e-5 || math.Abs(float64(vec[1])-0.8) > 1e-5 {
		t.Fatalf("vector %v", vec)
	}
	if math.Abs(norm(vec)-1) > 1e-5 {
		t.Fatalf("not normalized: %f", norm(vec))
	}
}
