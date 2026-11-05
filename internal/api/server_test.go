package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Below-Z3R0/argos/internal/service"
)

type fakeEngine struct{ last service.SearchRequest }

func (f *fakeEngine) Search(_ context.Context, req service.SearchRequest) (service.SearchResponse, error) {
	f.last = req
	return service.SearchResponse{Query: req.Query, Collection: "c", Hits: []service.Hit{{Path: "a.md", Score: 0.9}}}, nil
}

func (f *fakeEngine) Status(context.Context) service.Status {
	return service.Status{Collection: "c", Model: "embed"}
}

func (f *fakeEngine) Metrics() string { return "argos_search_total 1\n" }

func (f *fakeEngine) Ready(context.Context) error { return nil }

func (f *fakeEngine) Load(context.Context, string) error { return nil }

func (f *fakeEngine) Unload(string) {}

func TestSearchOK(t *testing.T) {
	fe := &fakeEngine{}
	srv := httptest.NewServer(New(fe).Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/search", "application/json", strings.NewReader(`{"query":"hola","db":"notas","k":3}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out service.SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Hits) != 1 || out.Hits[0].Path != "a.md" {
		t.Fatalf("out %+v", out)
	}
	if fe.last.DB != "notas" || fe.last.K != 3 {
		t.Fatalf("request not forwarded: %+v", fe.last)
	}
}

func TestSearchRejectsEmptyQuery(t *testing.T) {
	srv := httptest.NewServer(New(&fakeEngine{}).Handler())
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/v1/search", "application/json", strings.NewReader(`{"query":"  "}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestSearchRejectsBadJSON(t *testing.T) {
	srv := httptest.NewServer(New(&fakeEngine{}).Handler())
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/v1/search", "application/json", strings.NewReader(`{`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestSearchRejectsGET(t *testing.T) {
	srv := httptest.NewServer(New(&fakeEngine{}).Handler())
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/v1/search")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestStatusAndHealth(t *testing.T) {
	srv := httptest.NewServer(New(&fakeEngine{}).Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/status")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var st service.Status
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		t.Fatal(err)
	}
	if st.Model != "embed" {
		t.Fatalf("status %+v", st)
	}

	health, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer health.Body.Close()
	if health.StatusCode != http.StatusOK {
		t.Fatalf("health %d", health.StatusCode)
	}
}
