package service

import (
	"testing"

	"github.com/Below-Z3R0/argos/internal/types"
)

func testHit(path string, chunk int) types.Hit {
	return types.Hit{
		ID:    path,
		Score: 1,
		Payload: map[string]any{
			"path": path, "heading": "h", "text": "t",
			"chunk_index": int64(chunk), "tokens": int64(10),
		},
	}
}

func TestFuseHitsKeepsBothDatabases(t *testing.T) {
	targets := []Target{{Name: "a"}, {Name: "b"}}
	ranked := [][]types.Hit{
		{testHit("x.md", 0), testHit("y.md", 0)},
		{testHit("x.md", 0), testHit("z.md", 0)},
	}
	out := fuseHits(targets, ranked)
	if len(out) != 4 {
		t.Fatalf("got %d hits, want 4 (the same path in two bases is two results)", len(out))
	}
	// The two x.md hits are rank 1 in their base, so both must beat the rest.
	if out[0].Path != "x.md" || out[1].Path != "x.md" {
		t.Fatalf("top two = %q, %q; want x.md twice", out[0].Path, out[1].Path)
	}
	if out[0].DB == out[1].DB {
		t.Fatalf("both x.md hits came from %q", out[0].DB)
	}
	if out[0].Score != out[1].Score {
		t.Fatalf("equal ranks produced different scores: %v vs %v", out[0].Score, out[1].Score)
	}
	if out[2].Score >= out[0].Score {
		t.Fatalf("rank 2 scored %v, rank 1 scored %v", out[2].Score, out[0].Score)
	}
}

func TestFuseHitsToleratesShortRankings(t *testing.T) {
	out := fuseHits([]Target{{Name: "a"}, {Name: "b"}}, [][]types.Hit{{testHit("x.md", 0)}})
	if len(out) != 1 {
		t.Fatalf("got %d hits, want 1", len(out))
	}
}
