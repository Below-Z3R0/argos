package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultIsValid(t *testing.T) {
	cfg := Default()
	if cfg.Paths.Home == "" {
		t.Fatal("default home is empty")
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("default config invalid: %v", err)
	}
	if cfg.Defaults.Embedding == "" || cfg.Defaults.Reranker == "" {
		t.Fatal("defaults must pin one model per task")
	}
}

func TestLoadMissingReturnsDefaults(t *testing.T) {
	cfg, err := Load(filepath.Join(t.TempDir(), "nope.yaml"))
	if err != nil {
		t.Fatalf("missing file must not fail: %v", err)
	}
	if cfg.Resources.Batch != Default().Resources.Batch {
		t.Fatalf("expected defaults, got batch=%d", cfg.Resources.Batch)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "config.yaml")
	cfg := Default()
	cfg.Paths.Home = t.TempDir()
	cfg.Paths.Vault = t.TempDir()
	cfg.Defaults.Reranker = "qwen3-reranker-0.6b-q4"
	cfg.Resources.Mode = "cpu"
	if err := cfg.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.Defaults.Reranker != cfg.Defaults.Reranker || got.Resources.Mode != "cpu" {
		t.Fatalf("round trip mismatch: %+v", got)
	}
}

func TestValidateRejectsBadValues(t *testing.T) {
	cases := map[string]func(*Config){
		"relative home": func(c *Config) { c.Paths.Home = "relative/path" },
		"empty home":    func(c *Config) { c.Paths.Home = "" },
		"unknown mode":  func(c *Config) { c.Resources.Mode = "turbo" },
		"bad pause pct": func(c *Config) { c.Resources.PauseAtPct = 150 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := Default()
			cfg.Paths.Home = t.TempDir()
			mutate(&cfg)
			if err := cfg.Validate(); err == nil {
				t.Fatalf("expected validation error")
			}
		})
	}
}

func TestSaveIsAtomic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	cfg := Default()
	cfg.Paths.Home = dir
	if err := cfg.Save(path); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.Name() != "config.yaml" {
			t.Fatalf("leftover temp file: %s", e.Name())
		}
	}
}
