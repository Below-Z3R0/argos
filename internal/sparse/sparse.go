// Package sparse builds lexical (BM25-style) sparse vectors.
//
// Only term frequencies are emitted: Qdrant applies the IDF with
// `Modifier: Idf` on the collection. Emitting IDF here would apply it twice.
package sparse

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// Vector is a sparse representation ready for Qdrant.
type Vector struct {
	Indices []uint32
	Values  []float32
}

// Encode turns text into term frequencies. Tokens are lowercased, split on
// non-alphanumerics, filtered by length and a small stopword list, and hashed
// with FNV-1a so the same token always lands in the same index.
func Encode(text string) Vector {
	counts := map[uint32]int{}
	for _, tok := range tokens(text) {
		if len([]rune(tok)) < 2 || stopwords[tok] {
			continue
		}
		counts[hash(tok)]++
	}
	indices := make([]uint32, 0, len(counts))
	for idx := range counts {
		indices = append(indices, idx)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	values := make([]float32, len(indices))
	for i, idx := range indices {
		// Sublinear TF: repeated words matter, but not linearly.
		values[i] = float32(1 + math.Log(float64(counts[idx])))
	}
	return Vector{Indices: indices, Values: values}
}

func tokens(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func hash(s string) uint32 {
	const (
		offset = 2166136261
		prime  = 16777619
	)
	h := uint32(offset)
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= prime
	}
	return h
}

var stopwords = map[string]bool{
	"de": true, "la": true, "el": true, "los": true, "las": true, "un": true, "una": true,
	"y": true, "o": true, "que": true, "en": true, "con": true, "por": true, "para": true,
	"del": true, "al": true, "se": true, "es": true, "son": true, "como": true,
	"the": true, "a": true, "an": true, "of": true, "to": true, "in": true, "and": true,
	"or": true, "for": true, "on": true, "with": true, "is": true, "are": true, "as": true,
}
