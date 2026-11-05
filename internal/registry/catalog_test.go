package registry

import (
	"errors"
	"testing"
)

func TestEmbeddedCatalogLoads(t *testing.T) {
	c, err := Load()
	if err != nil {
		t.Fatalf("embedded catalogue must be valid: %v", err)
	}
	if len(c.Models) == 0 {
		t.Fatal("catalogue is empty")
	}
}

func TestDefaultPerTask(t *testing.T) {
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, task := range []string{TaskEmbedding, TaskReranker} {
		m, err := c.Default(task)
		if err != nil {
			t.Fatalf("no default for %s: %v", task, err)
		}
		if m.Task != task {
			t.Fatalf("default for %s has task %s", task, m.Task)
		}
	}
}

func TestByTaskOrdersBalancedFirst(t *testing.T) {
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	models := c.ByTask(TaskEmbedding)
	if len(models) < 2 {
		t.Skip("need at least two embeddings")
	}
	if models[0].Tier != "balanced" {
		t.Fatalf("first entry should be balanced, got %s", models[0].Tier)
	}
	if !models[0].Default {
		t.Fatalf("balanced entry should be the default")
	}
}

func TestGetUnknown(t *testing.T) {
	c, _ := Load()
	_, err := c.Get("does-not-exist")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}

func TestValidateRejects(t *testing.T) {
	base := Model{
		ID: "x", Name: "X", Task: TaskEmbedding, Tier: "balanced",
		Files: []File{{URL: "https://example.com/m.onnx", Dest: "model.onnx"}},
	}
	cases := map[string]func(*Catalog){
		"duplicate": func(c *Catalog) { c.Models = append(c.Models, c.Models[0]) },
		"bad task":  func(c *Catalog) { c.Models[0].Task = "vision" },
		"bad tier":  func(c *Catalog) { c.Models[0].Tier = "ultra" },
		"no files":  func(c *Catalog) { c.Models[0].Files = nil },
		"http url":  func(c *Catalog) { c.Models[0].Files[0].URL = "http://example.com/m.onnx" },
		"no dest":   func(c *Catalog) { c.Models[0].Files[0].Dest = "" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			c := Catalog{Models: []Model{base}}
			mutate(&c)
			if err := c.Validate(); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestStoreDir(t *testing.T) {
	if d, _ := StoreDir(TaskEmbedding); d != "embeddings" {
		t.Fatalf("got %s", d)
	}
	if d, _ := StoreDir(TaskReranker); d != "rerankers" {
		t.Fatalf("got %s", d)
	}
	if _, err := StoreDir("audio"); err == nil {
		t.Fatal("expected error for unknown task")
	}
}

func TestMergeOnlyTakesPins(t *testing.T) {
	base := Catalog{Models: []Model{{
		ID: "m", Name: "Model", Task: TaskEmbedding, Tier: "balanced", Default: true,
		Specs: Specs{RAMMB: 900},
		Files: []File{{URL: "https://example.com/m.onnx", Dest: "model.onnx"}},
	}}}
	base.Merge(Catalog{Models: []Model{{
		ID: "m", Name: "Old name", Task: TaskEmbedding, Tier: "light", Default: false,
		Specs: Specs{RAMMB: 1},
		Files: []File{{URL: "https://example.com/other.onnx", Dest: "model.onnx", SHA256: "abc"}},
	}}})
	got := base.Models[0]
	if got.Tier != "balanced" || !got.Default || got.Specs.RAMMB != 900 {
		t.Fatalf("stale fields leaked: %+v", got)
	}
	if got.Files[0].SHA256 != "abc" {
		t.Fatalf("pin not applied: %+v", got.Files[0])
	}
	if got.Files[0].URL != "https://example.com/m.onnx" {
		t.Fatalf("user file replaced the catalogue entry: %s", got.Files[0].URL)
	}
}
