// Package search turns a query into ranked passages.
//
// It owns the policy — how wide to retrieve, how many results per file, when to
// rerank — while the embedder, the store and the reranker stay replaceable.
package search

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/Below-Z3R0/argos/internal/sparse"
	"github.com/Below-Z3R0/argos/internal/types"
)

// Embedder embeds the query.
type Embedder interface {
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
}

// Store runs the vector search.
type Store interface {
	Search(ctx context.Context, collection string, vector []float32, limit int, filter map[string]string) ([]types.Hit, error)
}

// HybridStore adds the sparse+dense path. The Searcher only requires it when
// Hybrid is enabled, so a dense-only fake still works in tests.
type HybridStore interface {
	SearchHybrid(ctx context.Context, collection string, dense []float32, sparseIdx []uint32, sparseVal []float32, limit int, filter map[string]string) ([]types.Hit, error)
}

// Reranker scores (query, passage) pairs.
type Reranker interface {
	Score(ctx context.Context, query string, passages []string) ([]float32, error)
}

// Searcher is configured once and used per query.
type Searcher struct {
	Embedder    Embedder
	Store       Store
	Reranker    Reranker
	Collection  string
	K           int // passages returned
	Pool        int // candidates retrieved before dedupe/rerank
	MaxPerFile  int
	MinScore    float64
	RerankDepth int // candidates sent to the reranker
	Filter      map[string]string
	Hybrid      bool // dense+sparse with RRF; false is dense-only
}

// Result is one passage returned to the caller.
type Result struct {
	ID         string  `json:"id"`
	Path       string  `json:"path"`
	Heading    string  `json:"heading"`
	Text       string  `json:"text"`
	Score      float64 `json:"score"`
	Tokens     int     `json:"tokens"`
	ChunkIndex int     `json:"chunk_index"`
}

// Search runs the query and returns the best passages.
func (s Searcher) Search(ctx context.Context, query string) ([]Result, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("search: empty query")
	}
	if s.K <= 0 {
		s.K = 20
	}
	if s.Pool <= 0 {
		s.Pool = s.K * 3
	}
	if s.MaxPerFile <= 0 {
		s.MaxPerFile = 2
	}
	if s.RerankDepth <= 0 {
		s.RerankDepth = s.Pool
	}

	vec, err := s.Embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	var hits []types.Hit
	if s.Hybrid {
		hybrid, ok := s.Store.(HybridStore)
		if !ok {
			return nil, fmt.Errorf("search: el store no soporta búsqueda híbrida")
		}
		sv := sparse.Encode(query)
		hits, err = hybrid.SearchHybrid(ctx, s.Collection, vec, sv.Indices, sv.Values, s.Pool, s.Filter)
	} else {
		hits, err = s.Store.Search(ctx, s.Collection, vec, s.Pool, s.Filter)
	}
	if err != nil {
		return nil, err
	}
	results := toResults(hits)

	if s.Reranker != nil && len(results) > 0 {
		depth := min(s.RerankDepth, len(results))
		passages := make([]string, depth)
		for i := 0; i < depth; i++ {
			passages[i] = results[i].Text
		}
		scores, err := s.Reranker.Score(ctx, query, passages)
		if err != nil {
			return nil, fmt.Errorf("search: rerank: %w", err)
		}
		if len(scores) != depth {
			return nil, fmt.Errorf("search: reranker returned %d scores for %d passages", len(scores), depth)
		}
		for i := 0; i < depth; i++ {
			results[i].Score = float64(scores[i])
		}
		sort.SliceStable(results[:depth], func(i, j int) bool { return results[i].Score > results[j].Score })
	}

	out := make([]Result, 0, s.K)
	perFile := map[string]int{}
	for _, r := range results {
		if s.MinScore > 0 && r.Score < s.MinScore {
			continue
		}
		if perFile[r.Path] >= s.MaxPerFile {
			continue
		}
		perFile[r.Path]++
		out = append(out, r)
		if len(out) == s.K {
			break
		}
	}
	return out, nil
}

func toResults(hits []types.Hit) []Result {
	out := make([]Result, 0, len(hits))
	for _, h := range hits {
		out = append(out, Result{
			ID:         h.ID,
			Path:       str(h.Payload["path"]),
			Heading:    str(h.Payload["heading"]),
			Text:       str(h.Payload["text"]),
			Score:      float64(h.Score),
			Tokens:     num(h.Payload["tokens"]),
			ChunkIndex: num(h.Payload["chunk_index"]),
		})
	}
	return out
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func num(v any) int {
	switch n := v.(type) {
	case int64:
		return int(n)
	case int:
		return n
	case float64:
		return int(n)
	default:
		return 0
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
