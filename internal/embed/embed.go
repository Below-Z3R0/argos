// Package embed turns text into vectors.
//
// The interface hides the model family: callers never assemble prefixes or
// know which runtime runs underneath. That matters because a missing "query:"
// prefix degrades retrieval without producing an error.
package embed

import "context"

// Embedder is the contract used by indexing and search.
type Embedder interface {
	// Dim is the vector length. Bases are named after it, so it must be stable.
	Dim() int
	// Name identifies the model in logs and payloads.
	Name() string
	// MaxTokens is the longest input the model accepts.
	MaxTokens() int
	// EmbedPassages embeds documents. Batching is the caller's decision.
	EmbedPassages(ctx context.Context, texts []string) ([][]float32, error)
	// EmbedQuery embeds a search query, applying the family's query prefix.
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
}
