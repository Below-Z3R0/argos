package chunk

import (
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/Below-Z3R0/argos/internal/parse"
)

func words(s string) int { return len(strings.Fields(s)) }

func doc() parse.Document {
	return parse.Document{
		Rel:   "notas/docker.md",
		Title: "Notas de Docker",
		Sections: []parse.Section{
			{Level: 1, Heading: "Notas de Docker", Path: "Notas de Docker",
				Lines: []string{"Introducción al uso de contenedores en el servidor."}},
			{Level: 2, Heading: "Redes", Path: "Notas de Docker > Redes",
				Lines: []string{"Las redes bridge aíslan por defecto. Se publican puertos con -p."}},
		},
	}
}

func TestChunksCarryTitleAndHeading(t *testing.T) {
	s := Splitter{Counter: words, MaxTokens: 100, Overlap: 5, Namespace: uuid.NameSpaceURL}
	chunks := s.Split(doc())
	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}
	if !strings.HasPrefix(chunks[0].Text, "Notas de Docker") {
		t.Fatalf("missing title prefix: %q", chunks[0].Text)
	}
	last := chunks[len(chunks)-1]
	if !strings.Contains(last.Text, "Redes") {
		t.Fatalf("missing heading prefix: %q", last.Text)
	}
	if last.Heading != "Notas de Docker > Redes" {
		t.Fatalf("heading path %q", last.Heading)
	}
}

func TestRespectsMaxTokens(t *testing.T) {
	body := strings.Repeat("Una oración corta sobre indexado. ", 60)
	d := parse.Document{Rel: "a.md", Title: "A", Sections: []parse.Section{
		{Level: 1, Heading: "A", Path: "A", Lines: []string{body}},
	}}
	s := Splitter{Counter: words, MaxTokens: 40, Overlap: 5, Namespace: uuid.NameSpaceURL}
	for _, c := range s.Split(d) {
		if c.Tokens > 40 {
			t.Fatalf("chunk with %d tokens exceeds 40: %q", c.Tokens, c.Text[:60])
		}
	}
}

func TestOverlapRepeatsTheTail(t *testing.T) {
	// Each paragraph is exactly 10 words; budget 20 → two per chunk, overlap 10.
	var lines []string
	for i := 0; i < 6; i++ {
		lines = append(lines, "uno dos tres cuatro cinco seis siete ocho nueve diez")
	}
	d := parse.Document{Rel: "a.md", Sections: []parse.Section{
		{Level: 1, Heading: "S", Path: "S", Lines: []string{strings.Join(lines, "\n\n")}},
	}}
	s := Splitter{Counter: words, MaxTokens: 25, Overlap: 10, Namespace: uuid.NameSpaceURL}
	chunks := s.Split(d)
	if len(chunks) < 2 {
		t.Fatalf("expected several chunks, got %d", len(chunks))
	}
	firstPara := "uno dos tres cuatro cinco seis siete ocho nueve diez"
	if !strings.Contains(chunks[0].Text, firstPara) {
		t.Fatalf("first chunk lost its body")
	}
	if !strings.Contains(chunks[1].Text, firstPara) {
		t.Fatalf("second chunk should start with the overlap: %q", chunks[1].Text)
	}
}

func TestIDsAreDeterministic(t *testing.T) {
	s := Splitter{Counter: words, MaxTokens: 100, Overlap: 5, Namespace: uuid.NameSpaceURL}
	a := s.Split(doc())
	b := s.Split(doc())
	if len(a) != len(b) {
		t.Fatal("different lengths")
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Hash != b[i].Hash {
			t.Fatalf("chunk %d is not deterministic", i)
		}
	}
}

func TestEmptySectionIsSkipped(t *testing.T) {
	d := parse.Document{Rel: "a.md", Title: "A", Sections: []parse.Section{
		{Level: 1, Heading: "A", Path: "A", Lines: []string{"  "}},
	}}
	s := Splitter{Counter: words, MaxTokens: 50, Namespace: uuid.NameSpaceURL}
	if got := s.Split(d); len(got) != 0 {
		t.Fatalf("expected no chunks, got %d", len(got))
	}
}

func TestMergeShortJoinsSmallSections(t *testing.T) {
	ten := "uno dos tres cuatro cinco seis siete ocho nueve diez"
	d := parse.Document{Rel: "a.md", Title: "A", Sections: []parse.Section{
		{Level: 2, Heading: "S1", Path: "A > S1", Lines: []string{ten}},
		{Level: 2, Heading: "S2", Path: "A > S2", Lines: []string{ten}},
	}}
	s := Splitter{Counter: words, MaxTokens: 100, MinTokens: 30, Overlap: 5, Namespace: uuid.NameSpaceURL}
	chunks := s.Split(d)
	if len(chunks) != 1 {
		t.Fatalf("expected one merged chunk, got %d", len(chunks))
	}
	if !strings.Contains(chunks[0].Text, "S1") || !strings.Contains(chunks[0].Text, "S2") {
		t.Fatalf("merged chunk lost a section: %q", chunks[0].Text)
	}
}

func TestMergeNeverExceedsMaxTokens(t *testing.T) {
	ten := "uno dos tres cuatro cinco seis siete ocho nueve diez"
	var sections []parse.Section
	for i := 0; i < 6; i++ {
		sections = append(sections, parse.Section{Level: 2, Heading: "S", Path: "A > S", Lines: []string{ten}})
	}
	d := parse.Document{Rel: "a.md", Title: "A", Sections: sections}
	s := Splitter{Counter: words, MaxTokens: 40, MinTokens: 30, Overlap: 0, Namespace: uuid.NameSpaceURL}
	for _, c := range s.Split(d) {
		if c.Tokens > 40 {
			t.Fatalf("chunk with %d tokens exceeds 40", c.Tokens)
		}
	}
}

func TestDuplicateSectionsGetDifferentIDs(t *testing.T) {
	body := "Texto idéntico repetido en dos secciones."
	d := parse.Document{Rel: "a.md", Title: "A", Sections: []parse.Section{
		{Level: 2, Heading: "Uno", Path: "A > Uno", Lines: []string{body}},
		{Level: 2, Heading: "Dos", Path: "A > Dos", Lines: []string{body}},
	}}
	s := Splitter{Counter: words, MaxTokens: 100, MinTokens: 0, Namespace: uuid.NameSpaceURL}
	chunks := s.Split(d)
	if len(chunks) != 2 {
		t.Fatalf("chunks %d", len(chunks))
	}
	if chunks[0].ID == chunks[1].ID {
		t.Fatal("identical sections must not share a point id")
	}
}
