// Command embedspike embeds a fixed set of texts and dumps the vectors, so a
// second implementation (Python/onnxruntime) can be compared against it.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/Below-Z3R0/argos/internal/embed"
	"github.com/Below-Z3R0/argos/internal/registry"
)

func main() {
	model := flag.String("model", "", "ruta al model.onnx")
	tok := flag.String("tokenizer", "", "ruta al tokenizer.json")
	lib := flag.String("lib", "", "ruta a libonnxruntime.so")
	out := flag.String("out", "", "archivo JSON de salida")
	textFile := flag.String("text-file", "", "archivo con un texto por línea (reemplaza los de prueba)")
	dims := flag.Int("dims", 1024, "dimensión del vector")
	pooling := flag.String("pooling", "last_token", "last_token|mean|cls")
	prefix := flag.String("passage-prefix", "", "prefijo de pasaje (e5: 'passage: ')")
	batch := flag.Int("batch", 1, "tamaño de lote")
	flag.Parse()
	if *model == "" || *tok == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "faltan --model, --tokenizer o --out")
		os.Exit(2)
	}

	texts := []string{
		"El indexador guarda un manifiesto por base para no reindexar lo que no cambió.",
		"The reranker reorders the candidate pool before the agent sees it.",
		"Chunking by headings keeps the section title inside the passage.",
	}
	if *textFile != "" {
		raw, err := os.ReadFile(*textFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "text-file:", err)
			os.Exit(1)
		}
		texts = nil
		for _, line := range strings.Split(string(raw), "\n") {
			if strings.TrimSpace(line) != "" {
				texts = append(texts, line)
			}
		}
	}

	t0 := time.Now()
	e, err := embed.NewORT(embed.ORTOptions{
		ModelPath:     *model,
		TokenizerPath: *tok,
		LibraryPath:   *lib,
		Name:          "qwen3-embed-0.6b",
		Dims:          *dims,
		MaxTokens:     512,
		Pooling:       *pooling,
		Threads:       0,
		BatchSize:     *batch,
		Prefixes:      registryPrefixes(*prefix),
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer e.Close()

	t1 := time.Now()
	ctx := context.Background()
	vecs, err := e.EmbedPassages(ctx, texts)
	if err != nil {
		fmt.Fprintln(os.Stderr, "embed:", err)
		os.Exit(1)
	}

	for i := range vecs {
		for j := i + 1; j < len(vecs); j++ {
			fmt.Printf("cos(%d,%d) = %.6f\n", i, j, cosine(vecs[i], vecs[j]))
		}
	}
	t2 := time.Now()
	fmt.Printf("dim=%d n=%d load=%dms embed=%dms (%.0f ms/text)\n",
		len(vecs[0]), len(vecs), t1.Sub(t0).Milliseconds(), t2.Sub(t1).Milliseconds(),
		float64(t2.Sub(t1).Milliseconds())/float64(len(texts)))
	fmt.Printf("mem: %s\n", memStatus())

	payload := map[string]any{"texts": texts, "vectors": vecs}
	data, err := json.Marshal(payload)
	if err != nil {
		fmt.Fprintln(os.Stderr, "json:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, data, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(1)
	}
	fmt.Println("vectores en", *out)
}

// memStatus reports the process high-water mark from /proc.
func memStatus() string {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return "desconocida"
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			return "pico " + strings.Join(strings.Fields(line)[1:], " ")
		}
	}
	return "desconocida"
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
}

func registryPrefixes(passage string) registry.Prefixes {
	if passage == "" {
		return registry.Prefixes{}
	}
	return registry.Prefixes{Passage: passage}
}
