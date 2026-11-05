// Package config loads and persists the user configuration.
//
// The file is YAML, missing values fall back to defaults, and saving is
// atomic: a crash mid-write must never leave a truncated config.
package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config is the on-disk configuration.
type Config struct {
	Paths     Paths     `yaml:"paths"`
	Defaults  Defaults  `yaml:"defaults"`
	Filters   Filters   `yaml:"filters"`
	Chunking  Chunking  `yaml:"chunking"`
	Qdrant    Qdrant    `yaml:"qdrant"`
	Service   Service   `yaml:"service"`
	Layout    Layout    `yaml:"layout"`
	Index     Index     `yaml:"index"`
	Resources Resources `yaml:"resources"`
}

// Chunking controls passage size. The defaults follow the 400–512 token
// consensus: NVIDIA found 128 tokens worst and 512–1024 best, FloTorch found
// 512 the best end-to-end, and Anthropic's numbers assume 800.
type Chunking struct {
	MaxTokens int `yaml:"max_tokens"`
	MinTokens int `yaml:"min_tokens"`
	Overlap   int `yaml:"overlap"`
}

// Qdrant is the vector store endpoint.
type Qdrant struct {
	URL string `yaml:"url"` // host:port
}

// Service points at argosd. When reachable, the CLI uses it instead of loading
// the model itself.
type Service struct {
	URL string `yaml:"url"`
}

// Layout configures the optional docling-serve sidecar for OCR and complex
// PDF layouts.
type Layout struct {
	URL  string `yaml:"url"`
	Mode string `yaml:"mode"` // auto | always | never
}

// Index names the collection.
type Index struct {
	Collection string `yaml:"collection"`
	// Dims truncates Matryoshka embeddings (0 = the model's own size).
	// Changing it changes the vector space: it requires a new collection.
	Dims int `yaml:"dims"`
	// Quantization compresses stored vectors: none | int8.
	Quantization string `yaml:"quantization"`
}

// Filters decide what the indexer reads. The default is Markdown only: a vault
// full of binaries and attachments should not be indexed by accident.
type Filters struct {
	Include     []string `yaml:"include"`
	Exclude     []string `yaml:"exclude"`
	ExcludeDirs []string `yaml:"exclude_dirs"`
}

// Paths are the host directories Argos uses. State never lives inside the
// corpus: Home is a reserved directory, Vault is the read-only source.
type Paths struct {
	Home    string `yaml:"home"`
	Vault   string `yaml:"vault"`
	Compose string `yaml:"compose"`
}

// Defaults pins one model per task.
type Defaults struct {
	Embedding string `yaml:"embedding"`
	Reranker  string `yaml:"reranker"`
}

// Resources holds application-level tuning. Container limits are not here:
// they belong to Docker and are set through `argosctl resources`.
type Resources struct {
	Mode       string  `yaml:"mode"`
	Devices    Devices `yaml:"devices"`
	Threads    int     `yaml:"threads"`
	Batch      int     `yaml:"batch"`
	PauseAtPct int     `yaml:"pause_at_pct"`
}

// Devices chooses the execution provider per model. `auto` picks a discrete
// GPU when present, then the integrated one, then CPU.
type Devices struct {
	Embed  string `yaml:"embed"`
	Rerank string `yaml:"rerank"`
}

// Default returns the configuration used when no file exists.
func Default() Config {
	return Config{
		Paths: Paths{
			Home:    defaultHome(),
			Vault:   defaultVault(),
			Compose: defaultCompose(),
		},
		Defaults: Defaults{
			Embedding: "qwen3-embed-0.6b",
			Reranker:  "bge-reranker-v2-m3-int8",
		},
		Filters: Filters{
			Include:     []string{".md"},
			ExcludeDirs: []string{".git", ".obsidian", ".rag", ".trash", "node_modules", "venv", ".venv"},
		},
		Chunking:  Chunking{MaxTokens: 800, MinTokens: 500, Overlap: 120},
		Qdrant:    Qdrant{URL: firstNonEmpty(envOr("ARGOS_QDRANT_URL"), "127.0.0.1:6334")},
		Service:   Service{URL: firstNonEmpty(os.Getenv("ARGOS_SERVICE_URL"), "http://127.0.0.1:7712")},
		Layout:    Layout{URL: firstNonEmpty(os.Getenv("ARGOS_LAYOUT_URL"), "http://127.0.0.1:5001"), Mode: firstNonEmpty(os.Getenv("ARGOS_LAYOUT"), "auto")},
		Index:     Index{Collection: "argos_vault", Dims: 512, Quantization: "int8"},
		Resources: Resources{Mode: "auto", Devices: Devices{Embed: "auto", Rerank: "auto"}, Threads: 0, Batch: 1, PauseAtPct: 90},
	}
}

// FilePath is where the config lives.
func FilePath() (string, error) {
	if dir := os.Getenv("XDG_CONFIG_HOME"); dir != "" {
		return filepath.Join(dir, "argos", "config.yaml"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "argos", "config.yaml"), nil
}

// Load reads path. A missing file is not an error: it returns the defaults.
func Load(path string) (Config, error) {
	cfg := Default()
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return cfg, nil
	}
	if err != nil {
		return Config{}, fmt.Errorf("read config: %w", err)
	}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config %s: %w", path, err)
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// Save writes the config atomically, creating its directory.
func (c Config) Save(path string) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".config-*.yaml")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// Validate rejects values that would fail later with a confusing error.
func (c Config) Validate() error {
	if c.Paths.Home == "" {
		return errors.New("paths.home is required")
	}
	if !filepath.IsAbs(c.Paths.Home) {
		return fmt.Errorf("paths.home must be absolute, got %q", c.Paths.Home)
	}
	switch c.Resources.Mode {
	case "", "auto", "cpu", "gpu", "mixed":
	default:
		return fmt.Errorf("resources.mode %q is not auto|cpu|gpu|mixed", c.Resources.Mode)
	}
	if c.Qdrant.URL == "" {
		return errors.New("qdrant.url is required")
	}
	if c.Index.Collection == "" {
		return errors.New("index.collection is required")
	}
	switch c.Layout.Mode {
	case "", "auto", "always", "never":
	default:
		return fmt.Errorf("layout.mode %q is not auto|always|never", c.Layout.Mode)
	}
	switch c.Index.Quantization {
	case "", "none", "int8":
	default:
		return fmt.Errorf("index.quantization %q is not none|int8", c.Index.Quantization)
	}
	if c.Index.Dims < 0 {
		return fmt.Errorf("index.dims must be >= 0, got %d", c.Index.Dims)
	}
	for name, dev := range map[string]string{"embed": c.Resources.Devices.Embed, "rerank": c.Resources.Devices.Rerank} {
		switch dev {
		case "", "auto", "cpu", "gpu":
		default:
			return fmt.Errorf("resources.devices.%s %q is not auto|cpu|gpu", name, dev)
		}
	}
	if c.Chunking.MaxTokens > 0 {
		if c.Chunking.MinTokens < 0 || c.Chunking.MinTokens > c.Chunking.MaxTokens {
			return fmt.Errorf("chunking.min_tokens must be 0..max_tokens, got %d", c.Chunking.MinTokens)
		}
		if c.Chunking.Overlap < 0 || c.Chunking.Overlap >= c.Chunking.MaxTokens {
			return fmt.Errorf("chunking.overlap must be 0..max_tokens, got %d", c.Chunking.Overlap)
		}
	}
	if c.Resources.PauseAtPct < 0 || c.Resources.PauseAtPct > 100 {
		return fmt.Errorf("resources.pause_at_pct must be 0-100, got %d", c.Resources.PauseAtPct)
	}
	return nil
}

func defaultHome() string {
	if dir := os.Getenv("ARGOS_HOME"); dir != "" {
		return dir
	}
	if dir := os.Getenv("XDG_DATA_HOME"); dir != "" {
		return filepath.Join(dir, "argos")
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".local", "share", "argos")
	}
	return ""
}

// envOr reads an env override, stripping an http:// scheme where the config
// expects host:port.
func envOr(key string) string {
	v := strings.TrimSpace(os.Getenv(key))
	v = strings.TrimPrefix(v, "http://")
	v = strings.TrimPrefix(v, "https://")
	return v
}

func defaultVault() string {
	if v := os.Getenv("ARGOS_VAULT"); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, "Documents", "Knowledge")
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func defaultCompose() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, "Documents", "Docker", "argos")
}
