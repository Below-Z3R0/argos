// Package ingest turns files into points: parse, chunk, embed, upsert, commit.
//
// It is the only package that knows the order of the pipeline. The rest are
// independent pieces with narrow contracts, which is what makes the diff logic
// testable without a model or a vector database.
package ingest

import (
	"context"
	"fmt"
	"time"

	"github.com/Below-Z3R0/argos/internal/chunk"
	"github.com/Below-Z3R0/argos/internal/manifest"
	"github.com/Below-Z3R0/argos/internal/parse"
	"github.com/Below-Z3R0/argos/internal/parse/layout"
	"github.com/Below-Z3R0/argos/internal/types"
	"github.com/Below-Z3R0/argos/internal/vault"
)

// Embedder is the subset of the embedding model the pipeline needs.
type Embedder interface {
	Name() string
	Dim() int
	EmbedPassages(ctx context.Context, texts []string) ([][]float32, error)
}

// Store is the subset of Qdrant the pipeline needs.
type Store interface {
	EnsureCollection(ctx context.Context, name string, dim int, quantization string) error
	EnsurePayloadIndexes(ctx context.Context, name string) error
	UpsertWithWait(ctx context.Context, collection string, points []types.Point, wait bool) error
	DeleteByIDs(ctx context.Context, collection string, ids []string) error
	DeleteByPayload(ctx context.Context, collection, field, value string) error
}

// Options configures one run.
type Options struct {
	Collection   string
	Model        string
	Quantization string
	BatchSize    int    // points per upsert
	DryRun       bool   // compute the plan without touching the store
	Full         bool   // ignore the manifest and re-embed everything
	LayoutURL    string // docling-serve, empty disables it
	LayoutMode   string // auto | always | never
	Progress     func(Event)
}

// Event is one progress notification.
type Event struct {
	Rel     string
	Action  string // added, updated, deleted, skipped
	Chunks  int
	Embed   int
	Reused  int
	Current int
	Total   int
}

// Stats summarizes a run.
type Stats struct {
	Scanned int
	Skipped int
	Added   int
	Updated int
	Removed int
	Chunks  int
	Embed   int
	Reused  int
	Errors  []string
}

// Run indexes the files, updating only what changed.
func Run(
	ctx context.Context,
	files []vault.File,
	reg *parse.Registry,
	splitter chunk.Splitter,
	emb Embedder,
	store Store,
	man *manifest.Manifest,
	opts Options,
) (Stats, error) {
	if opts.BatchSize <= 0 {
		opts.BatchSize = 128
	}
	var stats Stats
	stats.Scanned = len(files)

	if !opts.DryRun {
		if err := store.EnsureCollection(ctx, opts.Collection, emb.Dim(), opts.Quantization); err != nil {
			return stats, err
		}
		if err := store.EnsurePayloadIndexes(ctx, opts.Collection); err != nil {
			return stats, err
		}
	}

	present := make(map[string]bool, len(files))
	for i, file := range files {
		present[file.Rel] = true
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		action, n, embedded, reused, err := indexFile(ctx, file, reg, splitter, emb, store, man, opts)
		if err != nil {
			stats.Errors = append(stats.Errors, fmt.Sprintf("%s: %v", file.Rel, err))
			continue
		}
		stats.Chunks += n
		stats.Embed += embedded
		stats.Reused += reused
		switch action {
		case "added":
			stats.Added++
		case "updated":
			stats.Updated++
		default:
			stats.Skipped++
		}
		if opts.Progress != nil {
			opts.Progress(Event{Rel: file.Rel, Action: action, Chunks: n, Embed: embedded, Reused: reused, Current: i + 1, Total: len(files)})
		}
	}

	removed, err := prune(ctx, present, store, man, opts)
	if err != nil {
		return stats, err
	}
	stats.Removed = removed
	return stats, nil
}

func indexFile(
	ctx context.Context,
	file vault.File,
	reg *parse.Registry,
	splitter chunk.Splitter,
	emb Embedder,
	store Store,
	man *manifest.Manifest,
	opts Options,
) (action string, chunks, embedded, reused int, err error) {
	hash, err := vault.Hash(file.Path)
	if err != nil {
		return "", 0, 0, 0, err
	}
	prev, known, err := man.Get(file.Rel)
	if err != nil {
		return "", 0, 0, 0, err
	}
	if opts.Full {
		prev, known = manifest.FileState{}, false
	}
	if known && prev.Hash == hash && prev.Model == opts.Model {
		return "skipped", 0, 0, 0, nil
	}

	if _, ok := reg.Lookup(file.Path); !ok {
		return "skipped", 0, 0, 0, nil
	}
	doc, err := reg.Parse(ctx, file.Path, file.Rel)
	if err != nil {
		return "", 0, 0, 0, err
	}
	if opts.LayoutURL != "" && activeLayout(opts.LayoutMode, doc) {
		if converted, cerr := layout.New(opts.LayoutURL).Convert(ctx, file.Path); cerr == nil {
			if parsed, perr := parse.MarkdownString(converted, file.Rel); perr == nil {
				doc = parsed
			}
		}
	}
	chunksList := splitter.Split(doc)
	if len(chunksList) == 0 {
		// Record the file so the next run skips it instead of parsing it again.
		if !opts.DryRun {
			_ = man.Put(manifest.FileState{Rel: file.Rel, Hash: hash, MTime: file.MTime, Size: file.Size, Model: opts.Model, Indexed: time.Now().UTC()})
		}
		return "skipped", 0, 0, 0, nil
	}

	desired := make(map[string]bool, len(chunksList))
	for _, c := range chunksList {
		desired[c.ID] = true
	}
	previous := make(map[string]bool, len(prev.Chunks))
	var stale []string
	for _, c := range prev.Chunks {
		previous[c.ID] = true
		if !desired[c.ID] {
			stale = append(stale, c.ID)
		}
	}

	// Only chunks that do not exist yet need an embedding: unchanged content
	// keeps its vector, which is the point of chunk-level diffing.
	var toEmbed []chunk.Chunk
	for _, c := range chunksList {
		if previous[c.ID] {
			reused++
			continue
		}
		toEmbed = append(toEmbed, c)
	}

	if opts.DryRun {
		return actionFor(known), len(chunksList), len(toEmbed), reused, nil
	}

	// A full reindex must not leave points from a previous chunking behind:
	// their ids may no longer exist in the new chunk set.
	if opts.Full {
		if err := store.DeleteByPayload(ctx, opts.Collection, "path", file.Rel); err != nil {
			return "", 0, 0, 0, err
		}
	}

	if len(toEmbed) > 0 {
		vecs, err := emb.EmbedPassages(ctx, textsOf(toEmbed))
		if err != nil {
			return "", 0, 0, 0, err
		}
		if len(vecs) != len(toEmbed) {
			return "", 0, 0, 0, fmt.Errorf("embedder returned %d vectors for %d chunks", len(vecs), len(toEmbed))
		}
		points := make([]types.Point, len(toEmbed))
		for i, c := range toEmbed {
			points[i] = types.Point{
				ID:     c.ID,
				Vector: vecs[i],
				Payload: map[string]any{
					"path":         file.Rel,
					"heading":      c.Heading,
					"chunk_index":  c.Index,
					"content_hash": c.Hash,
					"file_hash":    hash,
					"mtime":        file.MTime,
					"tokens":       c.Tokens,
					"text":         c.Text,
					"tags":         doc.Tags,
					"lang":         doc.Lang,
				},
			}
		}
		if err := upsertBatches(ctx, store, opts.Collection, points, opts.BatchSize); err != nil {
			return "", 0, 0, 0, err
		}
		embedded = len(toEmbed)
	}

	// Delete stale points only after the new ones are in. If the delete came
	// first and the upsert failed, the file would lose its content entirely.
	if len(stale) > 0 {
		if err := store.DeleteByIDs(ctx, opts.Collection, stale); err != nil {
			return "", 0, 0, 0, err
		}
	}

	refs := make([]manifest.ChunkRef, len(chunksList))
	for i, c := range chunksList {
		refs[i] = manifest.ChunkRef{ID: c.ID, Hash: c.Hash}
	}
	state := manifest.FileState{
		Rel:     file.Rel,
		Hash:    hash,
		MTime:   file.MTime,
		Size:    file.Size,
		Model:   opts.Model,
		Chunks:  refs,
		Indexed: time.Now().UTC(),
	}
	if err := man.Put(state); err != nil {
		return "", 0, 0, 0, err
	}
	return actionFor(known), len(chunksList), embedded, reused, nil
}

func upsertBatches(ctx context.Context, store Store, collection string, points []types.Point, size int) error {
	for start := 0; start < len(points); start += size {
		end := start + size
		if end > len(points) {
			end = len(points)
		}
		wait := end == len(points) // wait once, at the end
		if err := store.UpsertWithWait(ctx, collection, points[start:end], wait); err != nil {
			return err
		}
	}
	return nil
}

// prune removes points of files that no longer exist on disk.
func prune(ctx context.Context, present map[string]bool, store Store, man *manifest.Manifest, opts Options) (int, error) {
	all, err := man.All()
	if err != nil {
		return 0, err
	}
	removed := 0
	for _, st := range all {
		if present[st.Rel] {
			continue
		}
		if !opts.DryRun {
			if err := store.DeleteByPayload(ctx, opts.Collection, "file_hash", st.Hash); err != nil {
				return removed, err
			}
			if err := man.Delete(st.Rel); err != nil {
				return removed, err
			}
		}
		removed++
	}
	return removed, nil
}

// activeLayout decides whether a document goes through Docling. A PDF without
// a text layer always does: indexing an empty chunk is worse than paying the
// extra seconds.
func activeLayout(mode string, doc parse.Document) bool {
	switch mode {
	case "never":
		return false
	case "always":
		return true
	default:
		return doc.Meta["needs_ocr"] == "true"
	}
}

func actionFor(known bool) string {
	if known {
		return "updated"
	}
	return "added"
}

func textsOf(chunks []chunk.Chunk) []string {
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i] = c.Text
	}
	return out
}
