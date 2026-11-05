// Package chunk splits a document into passages that fit the embedding model.
//
// Splitting follows the document structure first (headings), then paragraphs,
// then sentences. Passages shorter than MinTokens are merged with the next
// ones: NVIDIA's chunking study found 128-token chunks performed worst while
// the 400–512 range performed best, and a note with many short sections would
// otherwise produce a pile of fragments.
package chunk

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/google/uuid"

	"github.com/Below-Z3R0/argos/internal/parse"
)

// Counter returns the number of tokens in a text. The indexer injects the
// tokenizer of the model being used, so chunk size follows the real limit.
type Counter func(string) int

// Splitter turns documents into chunks.
type Splitter struct {
	Counter   Counter
	MaxTokens int // hard limit per chunk
	MinTokens int // merge a chunk smaller than this with the next one
	Overlap   int // tokens repeated between consecutive chunks
	Namespace uuid.UUID
}

// Chunk is one indexable passage.
type Chunk struct {
	ID      string
	Text    string
	Heading string // breadcrumb of the section
	Index   int    // position inside the document
	Tokens  int
	Hash    string
}

// Split returns the chunks of a document, in order.
func (s Splitter) Split(doc parse.Document) []Chunk {
	if s.Counter == nil {
		s.Counter = func(t string) int { return len(strings.Fields(t)) }
	}
	if s.MaxTokens <= 0 {
		s.MaxTokens = 480
	}
	if s.Overlap < 0 {
		s.Overlap = 0
	}

	var raw []Chunk
	for _, section := range doc.Sections {
		prefix := header(doc.Title, section.Path)
		body := strings.TrimSpace(section.Body())
		if body == "" {
			continue
		}
		for _, text := range s.pack(prefix, body) {
			raw = append(raw, Chunk{Text: prefix + "\n\n" + text, Heading: section.Path})
		}
	}
	if s.MinTokens > 0 {
		raw = s.mergeShort(raw)
	}
	return s.finalize(doc.Rel, raw)
}

// mergeShort joins consecutive chunks while the result stays under MaxTokens
// and the current one is still below MinTokens.
func (s Splitter) mergeShort(chunks []Chunk) []Chunk {
	if len(chunks) == 0 {
		return chunks
	}
	out := make([]Chunk, 0, len(chunks))
	cur := chunks[0]
	curTokens := s.Counter(cur.Text)
	for _, next := range chunks[1:] {
		nextTokens := s.Counter(next.Text)
		if curTokens < s.MinTokens && curTokens+nextTokens <= s.MaxTokens {
			cur.Text += "\n\n" + next.Text
			curTokens += nextTokens
			continue
		}
		out = append(out, cur)
		cur, curTokens = next, nextTokens
	}
	return append(out, cur)
}

// finalize assigns the deterministic identity and position.
//
// The id is built from the path, the content hash and how many times that same
// hash appeared earlier in the document. Without the occurrence counter, two
// identical sections of one file would collapse into a single point.
func (s Splitter) finalize(rel string, chunks []Chunk) []Chunk {
	seen := map[string]int{}
	for i := range chunks {
		sum := sha256.Sum256([]byte(chunks[i].Text))
		chunks[i].Hash = hex.EncodeToString(sum[:])
		chunks[i].Index = i
		chunks[i].Tokens = s.Counter(chunks[i].Text)
		occurrence := seen[chunks[i].Hash]
		seen[chunks[i].Hash] = occurrence + 1
		seed := fmt.Sprintf("%s#%s#%d", rel, chunks[i].Hash, occurrence)
		chunks[i].ID = uuid.NewSHA1(s.Namespace, []byte(seed)).String()
	}
	return chunks
}

// header builds the context line placed above every chunk.
func header(title, sectionPath string) string {
	switch {
	case title == "" && sectionPath == "":
		return ""
	case sectionPath == "" || sectionPath == title:
		return title
	case title == "":
		return sectionPath
	default:
		return title + " > " + sectionPath
	}
}

// pack splits a section body into pieces that fit MaxTokens, with overlap.
func (s Splitter) pack(prefix, body string) []string {
	budget := s.MaxTokens - s.Counter(prefix+"\n\n")
	// A long heading path can eat most of the budget; keep a usable minimum
	// without ever exceeding the model limit.
	if budget < 20 {
		budget = 20
	}
	if budget > s.MaxTokens {
		budget = s.MaxTokens
	}
	pieces := splitPieces(body, budget, s.Counter)

	var out []string
	var current []string
	used := 0
	flush := func() {
		if len(current) == 0 {
			return
		}
		out = append(out, strings.TrimSpace(strings.Join(current, "\n\n")))
	}

	for _, piece := range pieces {
		n := s.Counter(piece)
		if used > 0 && used+n > budget {
			flush()
			current = overlapTail(current, s.Overlap, s.Counter)
			used = 0
			for _, p := range current {
				used += s.Counter(p)
			}
		}
		current = append(current, piece)
		used += n
	}
	flush()
	return out
}

// overlapTail keeps the last pieces whose total fits in the overlap budget.
func overlapTail(pieces []string, budget int, count Counter) []string {
	if budget <= 0 {
		return nil
	}
	total := 0
	start := len(pieces)
	for i := len(pieces) - 1; i >= 0; i-- {
		n := count(pieces[i])
		if total+n > budget {
			break
		}
		total += n
		start = i
	}
	if start == len(pieces) {
		return nil
	}
	return append([]string(nil), pieces[start:]...)
}

// splitPieces breaks a body into paragraphs, then sentences, then words.
func splitPieces(body string, budget int, count Counter) []string {
	var out []string
	for _, para := range splitParagraphs(body) {
		if count(para) <= budget {
			out = append(out, para)
			continue
		}
		for _, sentence := range splitSentences(para) {
			if count(sentence) <= budget {
				out = append(out, sentence)
				continue
			}
			out = append(out, splitWords(sentence, budget, count)...)
		}
	}
	return out
}

func splitParagraphs(body string) []string {
	var out []string
	for _, p := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n\n") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// splitSentences is deliberately simple: it only breaks on clear terminators
// followed by a space. Over-splitting a sentence is cheaper than emitting a
// chunk over the model limit.
func splitSentences(para string) []string {
	var out []string
	var b strings.Builder
	runes := []rune(para)
	for i, r := range runes {
		b.WriteRune(r)
		if r != '.' && r != '!' && r != '?' {
			continue
		}
		if i+1 < len(runes) && (runes[i+1] == ' ' || runes[i+1] == '\n') {
			out = append(out, strings.TrimSpace(b.String()))
			b.Reset()
		}
	}
	if rest := strings.TrimSpace(b.String()); rest != "" {
		out = append(out, rest)
	}
	return out
}

func splitWords(text string, budget int, count Counter) []string {
	var out []string
	var words []string
	for _, w := range strings.Fields(text) {
		words = append(words, w)
		if count(strings.Join(words, " ")) >= budget {
			out = append(out, strings.Join(words, " "))
			words = nil
		}
	}
	if len(words) > 0 {
		out = append(out, strings.Join(words, " "))
	}
	return out
}
