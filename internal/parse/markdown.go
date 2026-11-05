package parse

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// Markdown parses frontmatter and the heading hierarchy.
type Markdown struct{}

// Extensions implements Parser.
func (Markdown) Extensions() []string { return []string{".md", ".markdown"} }

// Parse reads a Markdown file into sections.
//
// The heading hierarchy becomes Section.Path, so a chunk can carry
// "Notas > Docker > Redes" even when the body never repeats it.
func (Markdown) Parse(ctx context.Context, path, rel string) (Document, error) {
	if err := ctx.Err(); err != nil {
		return Document{}, err
	}
	f, err := os.Open(path)
	if err != nil {
		return Document{}, err
	}
	defer f.Close()

	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return Document{}, fmt.Errorf("parse %s: %w", rel, err)
	}

	return parseMarkdownLines(lines, path, rel)
}

// MarkdownString parses Markdown already in memory, used by the layout/OCR
// converter which returns Markdown instead of writing a file.
func MarkdownString(text, rel string) (Document, error) {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	return parseMarkdownLines(lines, "", rel)
}

func parseMarkdownLines(lines []string, path, rel string) (Document, error) {
	if path == "" {
		path = rel
	}
	doc := Document{Path: path, Rel: rel, Meta: map[string]string{}}
	lines, meta := splitFrontmatter(lines)
	for k, v := range meta {
		doc.Meta[k] = fmt.Sprint(v)
	}
	if t, ok := meta["title"]; ok {
		doc.Title = fmt.Sprint(t)
	}
	if l, ok := meta["lang"]; ok {
		doc.Lang = fmt.Sprint(l)
	}
	doc.Tags = stringSlice(meta["tags"])

	doc.Sections = splitSections(lines)
	if doc.Title == "" {
		doc.Title = firstHeading(doc.Sections)
	}
	if doc.Title == "" {
		doc.Title = strings.TrimSuffix(filepath.Base(rel), filepath.Ext(rel))
	}
	return doc, nil
}

// splitFrontmatter removes a leading YAML block and returns its fields.
func splitFrontmatter(lines []string) ([]string, map[string]any) {
	meta := map[string]any{}
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return lines, meta
	}
	end := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			end = i
			break
		}
	}
	if end < 0 {
		return lines, meta
	}
	block := strings.Join(lines[1:end], "\n")
	_ = yaml.Unmarshal([]byte(block), &meta)
	return lines[end+1:], meta
}

// splitSections builds the heading tree. Content before the first heading
// becomes a level-0 section so nothing is silently dropped.
func splitSections(lines []string) []Section {
	var sections []Section
	var stack []Section // open headings, used for the breadcrumb
	var current *Section

	flush := func() {
		if current != nil {
			sections = append(sections, *current)
			current = nil
		}
	}

	start := func(level int, heading string) {
		flush()
		for len(stack) > 0 && stack[len(stack)-1].Level >= level {
			stack = stack[:len(stack)-1]
		}
		path := heading
		if len(stack) > 0 {
			path = stack[len(stack)-1].Path + " > " + heading
		}
		stack = append(stack, Section{Level: level, Heading: heading, Path: path})
		current = &Section{Level: level, Heading: heading, Path: path}
	}

	inFence := false
	var fenceMarker string
	for _, line := range lines {
		// A line starting with ``` or ~~~ toggles a code block. Inside it,
		// "# comentario" is shell code, not a heading.
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			marker := trimmed[:3]
			if !inFence {
				inFence, fenceMarker = true, marker
			} else if marker == fenceMarker {
				inFence, fenceMarker = false, ""
			}
			if current == nil {
				current = &Section{}
			}
			current.Lines = append(current.Lines, line)
			continue
		}
		if inFence {
			if current == nil {
				current = &Section{}
			}
			current.Lines = append(current.Lines, line)
			continue
		}
		if level, heading, ok := headingOf(line); ok {
			start(level, heading)
			continue
		}
		if current == nil {
			current = &Section{}
		}
		current.Lines = append(current.Lines, line)
	}
	flush()

	// Drop empty preamble sections, keep the rest in order.
	out := sections[:0]
	for _, s := range sections {
		if s.Level == 0 && strings.TrimSpace(s.Body()) == "" {
			continue
		}
		out = append(out, s)
	}
	return out
}

// headingOf returns the level and text of an ATX heading.
func headingOf(line string) (int, string, bool) {
	trimmed := strings.TrimLeft(line, " ")
	if len(line)-len(trimmed) > 3 {
		return 0, "", false // indented code block
	}
	level := 0
	for level < len(trimmed) && trimmed[level] == '#' {
		level++
	}
	if level == 0 || level > 6 || level >= len(trimmed) || trimmed[level] != ' ' {
		return 0, "", false
	}
	return level, strings.TrimSpace(trimmed[level+1:]), true
}

func firstHeading(sections []Section) string {
	for _, s := range sections {
		if s.Level == 1 && s.Heading != "" {
			return s.Heading
		}
	}
	return ""
}

func stringSlice(v any) []string {
	switch t := v.(type) {
	case []any:
		out := make([]string, 0, len(t))
		for _, x := range t {
			out = append(out, fmt.Sprint(x))
		}
		return out
	case string:
		if t == "" {
			return nil
		}
		return []string{t}
	default:
		return nil
	}
}
