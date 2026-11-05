// Package daemon talks to argosd, the process that keeps the models loaded.
package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client calls the Argos service.
type Client struct {
	BaseURL string
	HTTP    *http.Client
}

// New returns a client. An empty base URL disables it.
func New(baseURL string) *Client {
	if strings.TrimSpace(baseURL) == "" {
		return nil
	}
	return &Client{
		BaseURL: strings.TrimSuffix(baseURL, "/"),
		HTTP:    &http.Client{Timeout: 2 * time.Minute},
	}
}

// Search forwards a search request and returns the raw JSON response.
func (c *Client) Search(ctx context.Context, body any) ([]byte, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/search", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.do(req)
}

// Load asks the service to load a model (embed|rerank|all).
func (c *Client) Load(ctx context.Context, model string) ([]byte, error) {
	return c.post(ctx, "/v1/models/load", map[string]string{"model": model})
}

// Unload asks the service to free a model.
func (c *Client) Unload(ctx context.Context, model string) ([]byte, error) {
	return c.post(ctx, "/v1/models/unload", map[string]string{"model": model})
}

func (c *Client) post(ctx context.Context, path string, body any) ([]byte, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.do(req)
}

// Status returns the service status JSON.
func (c *Client) Status(ctx context.Context) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/v1/status", nil)
	if err != nil {
		return nil, err
	}
	return c.do(req)
}

func (c *Client) do(req *http.Request) ([]byte, error) {
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("argosd: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("argosd: %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	return raw, nil
}
