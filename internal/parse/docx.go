package parse

import (
	"archive/zip"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"regexp"
	"strings"
)

// DOCX reads the OOXML body and recovers the heading hierarchy from paragraph
// styles (Heading1..Heading6, and the Spanish ids Ttulo1..Ttulo6).
type DOCX struct{}

// Extensions implements Parser.
func (DOCX) Extensions() []string { return []string{".docx"} }

type docxDocument struct {
	Body struct {
		Paragraphs []docxParagraph `xml:"p"`
	} `xml:"body"`
}

type docxParagraph struct {
	Style struct {
		Val string `xml:"val,attr"`
	} `xml:"pPr>pStyle"`
	Runs []struct {
		Text string `xml:"t"`
	} `xml:"r"`
	// Text inside hyperlinks lives in a nested run list; without this, any
	// link in a Word document disappears from the index.
	Links []struct {
		Runs []struct {
			Text string `xml:"t"`
		} `xml:"r"`
	} `xml:"hyperlink"`
}

var headingStyle = regexp.MustCompile(`(?i)^(heading|t[ií]?tulo|title)\s*([1-6])?$`)

// Parse implements Parser.
func (DOCX) Parse(ctx context.Context, path, rel string) (Document, error) {
	if err := ctx.Err(); err != nil {
		return Document{}, err
	}
	zr, err := zip.OpenReader(path)
	if err != nil {
		return Document{}, err
	}
	defer zr.Close()

	var body io.ReadCloser
	for _, f := range zr.File {
		if f.Name == "word/document.xml" {
			body, err = f.Open()
			if err != nil {
				return Document{}, err
			}
			break
		}
	}
	if body == nil {
		return Document{}, fmt.Errorf("docx %s: missing word/document.xml", rel)
	}
	defer body.Close()

	var doc docxDocument
	if err := xml.NewDecoder(body).Decode(&doc); err != nil {
		return Document{}, fmt.Errorf("docx %s: %w", rel, err)
	}

	var sections []Section
	var stack []Section
	var current *Section
	flush := func() {
		if current != nil {
			sections = append(sections, *current)
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

	for _, p := range doc.Body.Paragraphs {
		var text strings.Builder
		for _, r := range p.Runs {
			text.WriteString(r.Text)
		}
		for _, link := range p.Links {
			for _, r := range link.Runs {
				text.WriteString(r.Text)
			}
		}
		value := strings.TrimSpace(text.String())
		if level, heading, ok := headingFromStyle(p.Style.Val); ok {
			start(level, firstNonEmpty(heading, value))
			continue
		}
		if value == "" {
			if current != nil {
				current.Lines = append(current.Lines, "")
			}
			continue
		}
		if current == nil {
			current = &Section{}
		}
		current.Lines = append(current.Lines, value)
	}
	flush()

	title := strings.TrimSuffix(baseName(rel), extOf(rel))
	if len(sections) > 0 && sections[0].Level == 1 && sections[0].Heading != "" {
		title = sections[0].Heading
	}
	return Document{Path: path, Rel: rel, Title: title, Meta: map[string]string{}, Sections: sections}, nil
}

func headingFromStyle(style string) (int, string, bool) {
	m := headingStyle.FindStringSubmatch(strings.TrimSpace(style))
	if m == nil {
		return 0, "", false
	}
	level := 1
	if len(m) > 2 && m[2] != "" {
		level = int(m[2][0] - '0')
	}
	if level < 1 || level > 6 {
		level = 1
	}
	return level, "", true
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}
