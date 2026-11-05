package qdrant

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

// Snapshot is one collection snapshot.
type Snapshot struct {
	Name      string `json:"name"`
	CreatedAt string `json:"creation_time"`
	Size      int64  `json:"size"`
	Checksum  string `json:"checksum,omitempty"`
}

// restBase is the HTTP endpoint. Qdrant serves REST on 6333 by convention,
// which is the port the compose publishes; when the deployment is remote or
// re-mapped, qdrant.rest_url overrides it.
func (s *Store) restBase() string {
	if s.restURL != "" {
		return s.restURL
	}
	return "http://" + s.host + ":6333"
}

func (s *Store) rest(ctx context.Context, method, path string, body any) ([]byte, error) {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, s.restBase()+path, reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("qdrant rest: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("qdrant rest: %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	return raw, nil
}

// CreateSnapshot writes a consistent snapshot of a collection.
func (s *Store) CreateSnapshot(ctx context.Context, collection string) (Snapshot, error) {
	raw, err := s.rest(ctx, http.MethodPost, "/collections/"+collection+"/snapshots", nil)
	if err != nil {
		return Snapshot{}, err
	}
	var out struct {
		Result Snapshot `json:"result"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return Snapshot{}, err
	}
	return out.Result, nil
}

// ListSnapshots returns the snapshots of a collection.
func (s *Store) ListSnapshots(ctx context.Context, collection string) ([]Snapshot, error) {
	raw, err := s.rest(ctx, http.MethodGet, "/collections/"+collection+"/snapshots", nil)
	if err != nil {
		return nil, err
	}
	var out struct {
		Result []Snapshot `json:"result"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out.Result, nil
}

// DeleteSnapshot removes a snapshot file.
func (s *Store) DeleteSnapshot(ctx context.Context, collection, name string) error {
	_, err := s.rest(ctx, http.MethodDelete, "/collections/"+collection+"/snapshots/"+name, nil)
	return err
}

// RestoreSnapshot recovers a collection from one of its snapshots.
//
// The file location is tried first: the collection-scoped URL stops working
// when the target collection no longer exists (404 on the snapshot download),
// while the node's own file keeps working and lets Qdrant recreate it.
func (s *Store) RestoreSnapshot(ctx context.Context, collection, name string) error {
	locations := []string{
		fmt.Sprintf("file://%s/%s/%s", s.snapshotsPath, collection, name),
		fmt.Sprintf("%s/collections/%s/snapshots/%s", s.restBase(), collection, name),
	}
	var lastErr error
	for _, location := range locations {
		_, err := s.rest(ctx, http.MethodPut, "/collections/"+collection+"/snapshots/recover",
			map[string]any{"location": location, "priority": "snapshot"})
		if err == nil {
			return nil
		}
		lastErr = err
	}
	return lastErr
}
