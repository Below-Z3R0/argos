package parse

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/net/html"
)

// HTML extracts headings (h1-h3) and text from an HTML document.
type HTML struct{}

// Extensions implements Parser.
func (HTML) Extensions() []string { return []string{".html", ".htm"} }

// Parse implements Parser.
func (HTML) Parse(ctx context.Context, path, rel string) (Document, error) {
	if err := ctx.Err(); err != nil {
		return Document{}, err
	}
	f, err := os.Open(path)
	if err != nil {
		return Document{}, err
	}
	defer f.Close()

	sections, err := sectionsFromHTML(f)
	if err != nil {
		return Document{}, fmt.Errorf("parse %s: %w", rel, err)
	}
	title := strings.TrimSuffix(baseName(rel), extOf(rel))
	if len(sections) > 0 && sections[0].Level == 1 && sections[0].Heading != "" {
		title = sections[0].Heading
	}
	return Document{
		Path:     path,
		Rel:      rel,
		Title:    title,
		Meta:     map[string]string{},
		Sections: sections,
	}, nil
}

// sectionsFromHTML walks the token stream, turning h1-h3 into sections.
// Script and style content is ignored: it is not prose.
func sectionsFromHTML(r io.Reader) ([]Section, error) {
	z := html.NewTokenizer(r)
	var sections []Section
	var stack []Section
	var current *Section
	inHeading := 0
	inSkip := 0
	var heading strings.Builder

	flush := func() {
		if current != nil {
			current.Lines = trimLines(current.Lines)
			if strings.TrimSpace(current.Body()) != "" || current.Heading != "" {
				sections = append(sections, *current)
			}
			current = nil
		}
	}
	start := func(level int, text string) {
		flush()
		for len(stack) > 0 && stack[len(stack)-1].Level >= level {
			stack = stack[:len(stack)-1]
		}
		path := text
		if len(stack) > 0 {
			path = stack[len(stack)-1].Path + " > " + text
		}
		stack = append(stack, Section{Level: level, Heading: text, Path: path})
		current = &Section{Level: level, Heading: text, Path: path}
	}

	for {
		tt := z.Next()
		switch tt {
		case html.ErrorToken:
			flush()
			if z.Err() != nil && z.Err().Error() != "EOF" {
				return sections, z.Err()
			}
			return sections, nil
		case html.StartTagToken:
			name, _ := z.TagName()
			tag := string(name)
			switch tag {
			case "script", "style":
				inSkip++
			case "h1", "h2", "h3", "h4", "h5", "h6":
				inHeading = int(tag[1] - '0')
				heading.Reset()
			case "p", "br", "div", "li", "tr":
				if current != nil {
					current.Lines = append(current.Lines, "")
				}
			}
		case html.EndTagToken:
			name, _ := z.TagName()
			tag := string(name)
			switch tag {
			case "script", "style":
				if inSkip > 0 {
					inSkip--
				}
			case "h1", "h2", "h3", "h4", "h5", "h6":
				if inHeading > 0 {
					start(inHeading, strings.TrimSpace(heading.String()))
					inHeading = 0
				}
			}
		case html.TextToken:
			if inSkip > 0 {
				continue
			}
			text := strings.TrimSpace(string(z.Text()))
			if text == "" {
				continue
			}
			if inHeading > 0 {
				heading.WriteString(" ")
				heading.WriteString(text)
				continue
			}
			if current == nil {
				current = &Section{}
			}
			current.Lines = append(current.Lines, text)
		}
	}
}

func trimLines(lines []string) []string {
	out := make([]string, 0, len(lines))
	blank := true
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l == "" {
			if blank {
				continue
			}
			blank = true
			out = append(out, "")
			continue
		}
		blank = false
		out = append(out, l)
	}
	for len(out) > 0 && out[len(out)-1] == "" {
		out = out[:len(out)-1]
	}
	return out
}

func baseName(rel string) string {
	if i := strings.LastIndex(rel, "/"); i >= 0 {
		return rel[i+1:]
	}
	return rel
}

func extOf(rel string) string {
	name := baseName(rel)
	if i := strings.LastIndex(name, "."); i >= 0 {
		return name[i:]
	}
	return ""
}
