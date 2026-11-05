package sparse

import "testing"

func TestEncodeIsDeterministic(t *testing.T) {
	a := Encode("indexado incremental de documentos")
	b := Encode("indexado incremental de documentos")
	if len(a.Indices) != len(b.Indices) {
		t.Fatal("different lengths")
	}
	for i := range a.Indices {
		if a.Indices[i] != b.Indices[i] || a.Values[i] != b.Values[i] {
			t.Fatal("not deterministic")
		}
	}
}

func TestStopwordsAndShortTokensIgnored(t *testing.T) {
	v := Encode("de la el un a x")
	if len(v.Indices) != 0 {
		t.Fatalf("expected empty, got %+v", v)
	}
}

func TestRepeatedTermWeighsMore(t *testing.T) {
	once := Encode("docker")
	twice := Encode("docker docker")
	if len(once.Values) != 1 || len(twice.Values) != 1 {
		t.Fatalf("once=%+v twice=%+v", once, twice)
	}
	if twice.Values[0] <= once.Values[0] {
		t.Fatalf("expected higher weight: %v vs %v", twice.Values[0], once.Values[0])
	}
	if twice.Indices[0] != once.Indices[0] {
		t.Fatal("same token must hash to the same index")
	}
}

func TestIndicesSorted(t *testing.T) {
	v := Encode("qdrant vector search embeddings retrieval docker linux")
	for i := 1; i < len(v.Indices); i++ {
		if v.Indices[i-1] >= v.Indices[i] {
			t.Fatalf("indices not sorted: %v", v.Indices)
		}
	}
}
