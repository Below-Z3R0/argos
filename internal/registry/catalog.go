// Package registry owns the model catalogue: what exists, what it costs, and
// which entry is the default for each task.
//
// The catalogue is data, not code. It ships embedded so the binary works
// offline, and a user file can override entries without recompiling.
package registry

import (
	_ "embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

//go:embed catalog.yaml
var embedded []byte

// Task values. A task is what the model does, and it also selects the store
// directory: embeddings/ or rerankers/.
const (
	TaskEmbedding = "embedding"
	TaskReranker  = "reranker"
)

// ErrNotFound is returned when an id is not in the catalogue.
var ErrNotFound = errors.New("model not found")

// Catalog is the parsed catalogue.
type Catalog struct {
	Version int     `yaml:"version"`
	Updated string  `yaml:"updated"`
	Models  []Model `yaml:"models"`
}

// Model is one catalogue entry.
type Model struct {
	ID       string `yaml:"id"`
	Name     string `yaml:"name"`
	Task     string `yaml:"task"`
	Tier     string `yaml:"tier"`
	Default  bool   `yaml:"default"`
	Family   string `yaml:"family"`
	License  string `yaml:"license"`
	Homepage string `yaml:"homepage"`
	Notes    string `yaml:"notes"`

	QueryInstruction string   `yaml:"query_instruction"`
	Prefixes         Prefixes `yaml:"prefixes"`
	Specs            Specs    `yaml:"specs"`
	Files            []File   `yaml:"files"`
}

// Prefixes are the asymmetric prefixes some families require (E5, Gemma).
type Prefixes struct {
	Query   string `yaml:"query"`
	Passage string `yaml:"passage"`
}

// Specs are the numbers a user needs to choose a model.
type Specs struct {
	Params        string `yaml:"params"`
	Dims          int    `yaml:"dims"`
	MaxTokens     int    `yaml:"max_tokens"`
	Pooling       string `yaml:"pooling"`
	Quant         string `yaml:"quant"`
	RAMMB         int    `yaml:"ram_mb"`
	DiskMB        int    `yaml:"disk_mb"`
	DecoderLayers int    `yaml:"decoder_layers"`
	Languages     int    `yaml:"languages"`
	Translation   bool   `yaml:"translation"`
	MRL           bool   `yaml:"mrl"`
}

// File is one downloadable artifact.
type File struct {
	URL       string `yaml:"url"`
	Dest      string `yaml:"dest"`
	SizeMB    int    `yaml:"size_mb"`
	SizeBytes int64  `yaml:"size_bytes"`
	SHA256    string `yaml:"sha256"`
}

// tierOrder decides how a list is shown: the entry most people should pick
// comes first, then the ones chosen for a specific reason.
var tierOrder = map[string]int{
	"balanced": 0,
	"fast":     1,
	"quality":  2,
	"compat":   3,
	"light":    4,
	"max":      5,
	"minimal":  6,
}

// Marshal renders a catalogue as YAML, used when writing user overrides.
func Marshal(c Catalog) ([]byte, error) {
	return yaml.Marshal(c)
}

// Load parses the embedded catalogue and validates it.
func Load() (*Catalog, error) {
	var c Catalog
	if err := yaml.Unmarshal(embedded, &c); err != nil {
		return nil, fmt.Errorf("parse embedded catalogue: %w", err)
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

// LoadFile parses an external catalogue and replaces embedded entries by id.
func LoadFile(path string) (*Catalog, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var user Catalog
	if err := yaml.Unmarshal(data, &user); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	base, err := Load()
	if err != nil {
		return nil, err
	}
	base.Merge(user)
	if err := base.Validate(); err != nil {
		return nil, err
	}
	return base, nil
}

// Merge applies user overrides.
//
// For an id that already exists, only the file pins (sha256 and size) are
// taken: a stale copy of the catalogue must not resurrect old tiers, RAM
// budgets or defaults. An unknown id is appended as a full entry.
func (c *Catalog) Merge(other Catalog) {
	for _, m := range other.Models {
		idx := -1
		for i := range c.Models {
			if c.Models[i].ID == m.ID {
				idx = i
				break
			}
		}
		if idx < 0 {
			c.Models = append(c.Models, m)
			continue
		}
		base := &c.Models[idx]
		for _, f := range m.Files {
			for j := range base.Files {
				if base.Files[j].Dest != f.Dest {
					continue
				}
				if f.SHA256 != "" {
					base.Files[j].SHA256 = f.SHA256
				}
				if f.SizeBytes != 0 {
					base.Files[j].SizeBytes = f.SizeBytes
				}
			}
		}
	}
}

// Validate catches catalogue mistakes at load time, not at install time.
func (c *Catalog) Validate() error {
	seen := map[string]bool{}
	for _, m := range c.Models {
		if m.ID == "" {
			return errors.New("catalogue entry without id")
		}
		if seen[m.ID] {
			return fmt.Errorf("duplicate model id %q", m.ID)
		}
		seen[m.ID] = true
		if m.Task != TaskEmbedding && m.Task != TaskReranker {
			return fmt.Errorf("model %q has unknown task %q", m.ID, m.Task)
		}
		if m.Name == "" {
			return fmt.Errorf("model %q has no name", m.ID)
		}
		if len(m.Files) == 0 {
			return fmt.Errorf("model %q has no files", m.ID)
		}
		if _, ok := tierOrder[m.Tier]; !ok {
			return fmt.Errorf("model %q has unknown tier %q", m.ID, m.Tier)
		}
		for _, f := range m.Files {
			if !strings.HasPrefix(f.URL, "https://") {
				return fmt.Errorf("model %q file %q must be https", m.ID, f.Dest)
			}
			if f.Dest == "" {
				return fmt.Errorf("model %q has a file without destination", m.ID)
			}
		}
	}
	return nil
}

// Get returns the entry with that id.
func (c *Catalog) Get(id string) (Model, error) {
	for _, m := range c.Models {
		if m.ID == id {
			return m, nil
		}
	}
	return Model{}, fmt.Errorf("%w: %s", ErrNotFound, id)
}

// Default returns the default entry for a task.
func (c *Catalog) Default(task string) (Model, error) {
	for _, m := range c.Models {
		if m.Task == task && m.Default {
			return m, nil
		}
	}
	return Model{}, fmt.Errorf("%w: no default for task %s", ErrNotFound, task)
}

// ByTask returns the models of a task ordered for display.
func (c *Catalog) ByTask(task string) []Model {
	var out []Model
	for _, m := range c.Models {
		if task == "" || m.Task == task {
			out = append(out, m)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Task != out[j].Task {
			return out[i].Task < out[j].Task
		}
		ti, tj := tierOrder[out[i].Tier], tierOrder[out[j].Tier]
		if ti != tj {
			return ti < tj
		}
		return out[i].Specs.RAMMB < out[j].Specs.RAMMB
	})
	return out
}

// Tasks lists the tasks present in the catalogue.
func (c *Catalog) Tasks() []string {
	seen := map[string]bool{}
	var out []string
	for _, m := range c.Models {
		if !seen[m.Task] {
			seen[m.Task] = true
			out = append(out, m.Task)
		}
	}
	sort.Strings(out)
	return out
}

// StoreDir maps a task to its directory under the models root.
func StoreDir(task string) (string, error) {
	switch task {
	case TaskEmbedding:
		return "embeddings", nil
	case TaskReranker:
		return "rerankers", nil
	default:
		return "", fmt.Errorf("unknown task %q", task)
	}
}

// UserCatalogPath is the optional override file.
func UserCatalogPath() (string, error) {
	if dir := os.Getenv("XDG_CONFIG_HOME"); dir != "" {
		return filepath.Join(dir, "argos", "models.yaml"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "argos", "models.yaml"), nil
}
