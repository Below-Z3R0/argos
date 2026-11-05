// Package layout talks to a docling-serve instance for PDFs that need layout
// analysis or OCR. It is optional: without the sidecar the pipeline falls back
// to pdftotext and marks the file.
package layout

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Client converts files through docling-serve.
type Client struct {
	BaseURL string
	HTTP    *http.Client
}

// New returns a client for a docling-serve base URL (http://host:5001).
func New(baseURL string) *Client {
	return &Client{BaseURL: strings.TrimSuffix(baseURL, "/"), HTTP: &http.Client{Timeout: 10 * time.Minute}}
}

// Convert uploads a file and returns Markdown with the document structure.
func (c *Client) Convert(ctx context.Context, path string) (string, error) {
	if c.BaseURL == "" {
		return "", fmt.Errorf("layout: no docling URL configured")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	part, err := mw.CreateFormFile("files", filepath.Base(path))
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(part, f); err != nil {
		return "", err
	}
	for k, v := range map[string]string{
		"to_formats":        "md",
		"do_ocr":            "true",
		"table_mode":        "fast",
		"image_export_mode": "placeholder",
	} {
		if err := mw.WriteField(k, v); err != nil {
			return "", err
		}
	}
	if err := mw.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/convert/file", &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("Accept", "application/json, text/markdown")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return "", fmt.Errorf("layout: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("layout: docling returned %s: %s", resp.Status, truncate(string(raw), 200))
	}
	if strings.Contains(resp.Header.Get("Content-Type"), "markdown") || strings.Contains(resp.Header.Get("Content-Type"), "text/plain") {
		return string(raw), nil
	}

	// The JSON shape has changed across versions; accept the documented
	// document.md_content and a top-level md_content.
	var payload struct {
		Document struct {
			MDContent string `json:"md_content"`
		} `json:"document"`
		MDContent string `json:"md_content"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return "", fmt.Errorf("layout: unexpected response: %w", err)
	}
	if payload.Document.MDContent != "" {
		return payload.Document.MDContent, nil
	}
	if payload.MDContent != "" {
		return payload.MDContent, nil
	}
	return "", fmt.Errorf("layout: response has no markdown content")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
