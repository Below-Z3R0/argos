// Command argosctl is the host-side CLI for Argos.
//
// It runs on the machine that owns the corpus: it creates databases, installs
// models, and reads container state. The service itself (argosd) runs in the
// container; this binary talks to it over HTTP.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/Below-Z3R0/argos/internal/chunk"
	"github.com/Below-Z3R0/argos/internal/config"
	dbpkg "github.com/Below-Z3R0/argos/internal/db"
	"github.com/Below-Z3R0/argos/internal/embed"
	"github.com/Below-Z3R0/argos/internal/ingest"
	"github.com/Below-Z3R0/argos/internal/manifest"
	"github.com/Below-Z3R0/argos/internal/models"
	"github.com/Below-Z3R0/argos/internal/parse"
	"github.com/Below-Z3R0/argos/internal/registry"
	"github.com/Below-Z3R0/argos/internal/rerank"
	"github.com/Below-Z3R0/argos/internal/resources"
	"github.com/Below-Z3R0/argos/internal/search"
	qstore "github.com/Below-Z3R0/argos/internal/store/qdrant"
	"github.com/Below-Z3R0/argos/internal/vault"
	"github.com/Below-Z3R0/argos/internal/version"
	"github.com/daulet/tokenizers"
	"github.com/google/uuid"
	"gopkg.in/yaml.v3"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "argosctl:", err)
		os.Exit(exitCode(err))
	}
}

func exitCode(err error) int {
	switch {
	case errors.Is(err, errUsage):
		return 1
	case errors.Is(err, registry.ErrNotFound), errors.Is(err, models.ErrNotInstalled):
		return 2
	default:
		return 2
	}
}

var errUsage = errors.New("usage")

func run(args []string) error {
	if len(args) == 0 {
		usage()
		return errUsage
	}
	switch args[0] {
	case "version", "--version", "-v":
		fmt.Println(version.String())
		return nil
	case "help", "-h", "--help":
		usage()
		return nil
	case "init":
		return cmdInit(args[1:])
	case "paths":
		return cmdPaths(args[1:])
	case "models":
		return cmdModels(args[1:])
	case "config":
		return cmdConfig(args[1:])
	case "index":
		return cmdIndex(args[1:])
	case "db":
		return cmdDB(args[1:])
	case "search":
		return cmdSearch(args[1:])
	case "service":
		return cmdService(args[1:])
	case "snapshot":
		return cmdSnapshot(args[1:])
	case "completion":
		return cmdCompletion(args[1:])
	case "eval":
		return cmdEval(args[1:])
	case "doctor":
		return cmdDoctor(args[1:])
	default:
		usage()
		return fmt.Errorf("%w: unknown command %q", errUsage, args[0])
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `argosctl — control de Argos

Uso:
  argosctl <comando> [flags]

Comandos:
  init        crea la raíz de estado, la config y el .env del stack
  paths       muestra las rutas en uso
  models      lista, instala y elige modelos (list|catalog|show|install|use|remove|verify|sync)
  config      path|list|get|set (rutas, defaults y filtros)
  index       --dry-run: qué archivos entrarían con los filtros actuales
  db          crea y administra bases independientes (create|list|info|rm)
  search      busca en una base (con reranker opcional)
  service     estado y carga/descarga de modelos en argosd (status|load|unload)
  snapshot    crea, lista, borra o restaura snapshots de una base
  completion  script de autocompletado (bash|zsh|fish)
  eval        mide recall@k/MRR contra un set de consultas
  doctor      verifica entorno, rutas y modelos
  version     versión del binario
`)
}

// --- helpers ---------------------------------------------------------------

func loadConfig() (config.Config, string, error) {
	path, err := config.FilePath()
	if err != nil {
		return config.Config{}, "", err
	}
	cfg, err := config.Load(path)
	return cfg, path, err
}

func openCatalog() (*registry.Catalog, error) {
	user, err := registry.UserCatalogPath()
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(user); err == nil {
		return registry.LoadFile(user)
	}
	return registry.Load()
}

func taskOrAll(v string) (string, error) {
	switch v {
	case "", "all":
		return "", nil
	case registry.TaskEmbedding, registry.TaskReranker:
		return v, nil
	default:
		return "", fmt.Errorf("%w: --task %q", errUsage, v)
	}
}

// --- init / paths ----------------------------------------------------------

func cmdInit(args []string) error {
	fs := flag.NewFlagSet("init", flag.ContinueOnError)
	home := fs.String("home", "", "raíz de estado (default: la de la config)")
	vault := fs.String("vault", "", "ruta del corpus")
	noCompose := fs.Bool("no-compose", false, "no escribir el .env del stack")
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, cfgPath, err := loadConfig()
	if err != nil {
		return err
	}
	if *home != "" {
		cfg.Paths.Home = *home
	}
	if *vault != "" {
		cfg.Paths.Vault = *vault
	}
	if err := cfg.Validate(); err != nil {
		return err
	}

	for _, sub := range []string{"data", "models", "qdrant", "docling"} {
		if err := os.MkdirAll(filepath.Join(cfg.Paths.Home, sub), 0o755); err != nil {
			return err
		}
	}
	for _, sub := range []string{"embeddings", "rerankers"} {
		if err := os.MkdirAll(filepath.Join(cfg.Paths.Home, "models", sub), 0o755); err != nil {
			return err
		}
	}
	if err := cfg.Save(cfgPath); err != nil {
		return err
	}
	fmt.Printf("config   %s\n", cfgPath)
	fmt.Printf("home     %s\n", cfg.Paths.Home)
	fmt.Printf("vault    %s\n", cfg.Paths.Vault)

	if !*noCompose && cfg.Paths.Compose != "" {
		if err := writeEnv(cfg); err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				fmt.Fprintf(os.Stderr, "aviso: no se pudo escribir el .env: %v\n", err)
			}
		} else {
			fmt.Printf("env      %s\n", filepath.Join(cfg.Paths.Compose, ".env"))
		}
	}
	return nil
}

func writeEnv(cfg config.Config) error {
	if _, err := os.Stat(cfg.Paths.Compose); err != nil {
		return err
	}
	body := fmt.Sprintf("# Generado por `argosctl init`. No editar a mano.\nARGOS_HOME=%s\nVAULT_PATH=%s\n",
		cfg.Paths.Home, cfg.Paths.Vault)
	path := filepath.Join(cfg.Paths.Compose, ".env")
	tmp, err := os.CreateTemp(cfg.Paths.Compose, ".env-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.WriteString(body); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

func cmdPaths(args []string) error {
	fs := flag.NewFlagSet("paths", flag.ContinueOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, cfgPath, err := loadConfig()
	if err != nil {
		return err
	}
	fmt.Printf("config   %s\n", cfgPath)
	fmt.Printf("home     %s\n", cfg.Paths.Home)
	fmt.Printf("models   %s\n", filepath.Join(cfg.Paths.Home, "models"))
	fmt.Printf("data     %s\n", filepath.Join(cfg.Paths.Home, "data"))
	fmt.Printf("qdrant   %s\n", filepath.Join(cfg.Paths.Home, "qdrant"))
	fmt.Printf("vault    %s\n", cfg.Paths.Vault)
	fmt.Printf("compose  %s\n", cfg.Paths.Compose)
	return nil
}

// --- models ----------------------------------------------------------------

func cmdModels(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("%w: models <list|catalog|show|install|use|remove|verify|sync>", errUsage)
	}
	switch args[0] {
	case "list":
		return modelsList(args[1:])
	case "catalog":
		return modelsCatalog(args[1:])
	case "show":
		return modelsShow(args[1:])
	case "install":
		return modelsInstall(args[1:], false)
	case "use":
		return modelsInstall(args[1:], true)
	case "remove":
		return modelsRemove(args[1:])
	case "verify":
		return modelsVerify(args[1:])
	case "sync":
		return modelsSync(args[1:])
	default:
		return fmt.Errorf("%w: models %s", errUsage, args[0])
	}
}

func newStore() (*models.Store, config.Config, error) {
	cfg, _, err := loadConfig()
	if err != nil {
		return nil, config.Config{}, err
	}
	return models.New(filepath.Join(cfg.Paths.Home, "models")), cfg, nil
}

func modelsList(args []string) error {
	fs := flag.NewFlagSet("models list", flag.ContinueOnError)
	task := fs.String("task", "", "embedding|reranker")
	all := fs.Bool("all", false, "incluye los no instalados")
	if err := fs.Parse(args); err != nil {
		return err
	}
	t, err := taskOrAll(*task)
	if err != nil {
		return err
	}
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	store, cfg, err := newStore()
	if err != nil {
		return err
	}
	mem := memoryStates(cfg)
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "TAREA\tID\tRAM\tDISCO\tINSTALADO\tDEFAULT\tMEMORIA")
	empty := true
	for _, m := range cat.ByTask(t) {
		_, ok := store.Status(m.Task, m.ID)
		installed := "sí"
		if !ok {
			installed = "no"
			if dl, exists := store.State(m.Task, m.ID); exists {
				switch dl.State {
				case "downloading":
					pct := 0.0
					if dl.TotalBytes > 0 {
						pct = float64(dl.DoneBytes) / float64(dl.TotalBytes) * 100
					}
					installed = fmt.Sprintf("descargando %.0f%%", pct)
				case "failed":
					installed = "incompleta"
				}
			}
		}
		if !ok && !*all && installed == "no" {
			continue
		}
		empty = false
		def := "—"
		if (m.Task == registry.TaskEmbedding && cfg.Defaults.Embedding == m.ID) ||
			(m.Task == registry.TaskReranker && cfg.Defaults.Reranker == m.ID) {
			def = "sí"
		}
		memLabel := mem[m.ID]
		if memLabel == "" {
			memLabel = "—"
		}
		fmt.Fprintf(w, "%s\t%s\t%d MB\t%d MB\t%s\t%s\t%s\n",
			m.Task, m.ID, m.Specs.RAMMB, m.Specs.DiskMB, installed, def, memLabel)
	}
	w.Flush()
	if empty {
		fmt.Println("(sin modelos instalados; usá `argosctl models catalog`)")
	}
	return nil
}

func state(m registry.Model, installed bool, cfg config.Config) string {
	var parts []string
	if installed {
		parts = append(parts, "instalado")
	} else {
		parts = append(parts, "no instalado")
	}
	if (m.Task == registry.TaskEmbedding && cfg.Defaults.Embedding == m.ID) ||
		(m.Task == registry.TaskReranker && cfg.Defaults.Reranker == m.ID) {
		parts = append(parts, "default")
	}
	return strings.Join(parts, " · ")
}

func modelsCatalog(args []string) error {
	fs := flag.NewFlagSet("models catalog", flag.ContinueOnError)
	task := fs.String("task", "", "embedding|reranker")
	tier := fs.String("tier", "", "filtra por tier")
	if err := fs.Parse(args); err != nil {
		return err
	}
	t, err := taskOrAll(*task)
	if err != nil {
		return err
	}
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	store, _, err := newStore()
	if err != nil {
		return err
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "TIER\tID\tPARAMS\tDIMS/CAPAS\tRAM\tDISCO\tLICENCIA\tINSTALADO\tDEFAULT")
	for _, m := range cat.ByTask(t) {
		if *tier != "" && m.Tier != *tier {
			continue
		}
		_, ok := store.Status(m.Task, m.ID)
		size := fmt.Sprintf("%dx%d", m.EffectiveDims(), m.Specs.MaxTokens)
		if m.Task == registry.TaskReranker {
			size = fmt.Sprintf("%d capas", m.Specs.DecoderLayers)
			if m.Specs.MaxTokens > 0 {
				size = fmt.Sprintf("%d tok", m.Specs.MaxTokens)
			}
		}
		label := m.Tier
		if m.Default {
			label += "*"
		}
		installed := "sí"
		if !ok {
			installed = "no"
		}
		def := "—"
		if m.Default {
			def = "sí"
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d MB\t%d MB\t%s\t%s\t%s\n",
			label, m.ID, m.Specs.Params, size, m.Specs.RAMMB, m.Specs.DiskMB, m.License, installed, def)
	}
	w.Flush()
	fmt.Println("\n* = default de su tarea. Ficha completa: argosctl models show <id>")
	return nil
}

func modelsShow(args []string) error {
	fs := flag.NewFlagSet("models show", flag.ContinueOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("%w: models show <id>", errUsage)
	}
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	m, err := cat.Get(fs.Arg(0))
	if err != nil {
		return err
	}
	store, cfg, err := newStore()
	if err != nil {
		return err
	}
	rec, ok := store.Status(m.Task, m.ID)
	fmt.Printf("%s (%s)\n", m.Name, m.ID)
	fmt.Printf("  tarea      %s · tier %s\n", m.Task, m.Tier)
	fmt.Printf("  params     %s\n", m.Specs.Params)
	if m.Specs.Dims > 0 {
		extra := ""
		if m.EffectiveDims() != m.Specs.Dims {
			extra = fmt.Sprintf(" (default %d)", m.EffectiveDims())
		}
		fmt.Printf("  dims       %d%s\n", m.Specs.Dims, extra)
	}
	if m.Specs.DecoderLayers > 0 {
		fmt.Printf("  capas      %d\n", m.Specs.DecoderLayers)
	}
	if m.Specs.MaxTokens > 0 {
		fmt.Printf("  contexto   %d tokens\n", m.Specs.MaxTokens)
	}
	fmt.Printf("  RAM        %d MB (presupuesto)\n", m.Specs.RAMMB)
	fmt.Printf("  disco      %d MB\n", m.Specs.DiskMB)
	fmt.Printf("  licencia   %s\n", m.License)
	fmt.Printf("  oficial    %s\n", m.Homepage)
	if m.Notes != "" {
		fmt.Printf("  notas      %s\n", m.Notes)
	}
	if ok {
		fmt.Printf("  instalado  sí · %s · %d bytes\n", rec.InstalledAt.Format(time.RFC3339), rec.SizeBytes)
	} else {
		fmt.Printf("  instalado  no\n")
	}
	isDefault := (m.Task == registry.TaskEmbedding && cfg.Defaults.Embedding == m.ID) ||
		(m.Task == registry.TaskReranker && cfg.Defaults.Reranker == m.ID)
	def := "no"
	if isDefault {
		def = "sí"
	}
	fmt.Printf("  default    %s\n", def)
	fmt.Println("  archivos:")
	for _, f := range m.Files {
		pin := "sin pin"
		if f.SHA256 != "" {
			pin = f.SHA256[:12] + "…"
		}
		fmt.Printf("    %-14s %6d MB  %s  %s\n", f.Dest, f.SizeMB, pin, f.URL)
	}
	return nil
}

func modelsInstall(args []string, use bool) error {
	fs := flag.NewFlagSet("models install", flag.ContinueOnError)
	yes := fs.Bool("yes", false, "no pedir confirmación")
	allow := fs.Bool("allow-unpinned", false, "instalar aunque falte sha256")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("%w: falta el id del modelo", errUsage)
	}
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	store, cfg, err := newStore()
	if err != nil {
		return err
	}

	for _, id := range fs.Args() {
		m, err := cat.Get(id)
		if err != nil {
			return err
		}
		if _, ok := store.Status(m.Task, m.ID); !ok {
			fmt.Printf("%s: %d MB desde %s (%s)\n", m.ID, m.Specs.DiskMB, m.Files[0].URL, m.License)
			if !*yes {
				fmt.Print("¿instalar? [y/N] ")
				var answer string
				fmt.Scanln(&answer)
				if strings.ToLower(strings.TrimSpace(answer)) != "y" {
					return fmt.Errorf("%w: instalación cancelada", errUsage)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
			last := time.Now()
			err = store.Install(ctx, m, models.Options{
				AllowUnpinned: *allow,
				Progress: func(done, total int64) {
					if time.Since(last) < time.Second && done < total {
						return
					}
					pct := 0.0
					if total > 0 {
						pct = float64(done) / float64(total) * 100
					}
					fmt.Printf("\r  %s: %3.0f%%  %.0f/%.0f MB", m.ID, pct, float64(done)/(1<<20), float64(total)/(1<<20))
					last = time.Now()
				},
			})
			cancel()
			if err != nil {
				return err
			}
			fmt.Printf("\r  %s: instalado\n", m.ID)
		} else {
			fmt.Printf("%s: ya instalado\n", m.ID)
		}

		if use {
			if err := setDefault(&cfg, m); err != nil {
				return err
			}
		}
	}
	if use {
		cfgPath, err := config.FilePath()
		if err != nil {
			return err
		}
		if err := cfg.Save(cfgPath); err != nil {
			return err
		}
		fmt.Printf("default %s: %s\n", catModelTask(cat, fs.Arg(0)), fs.Arg(0))
	}
	return nil
}

func catModelTask(cat *registry.Catalog, id string) string {
	if m, err := cat.Get(id); err == nil {
		return m.Task
	}
	return "modelo"
}

func setDefault(cfg *config.Config, m registry.Model) error {
	switch m.Task {
	case registry.TaskEmbedding:
		cfg.Defaults.Embedding = m.ID
	case registry.TaskReranker:
		cfg.Defaults.Reranker = m.ID
	default:
		return fmt.Errorf("unknown task %q", m.Task)
	}
	return nil
}

func modelsRemove(args []string) error {
	fs := flag.NewFlagSet("models remove", flag.ContinueOnError)
	force := fs.Bool("force", false, "borrar aunque sea el default")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("%w: models remove <id>", errUsage)
	}
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	m, err := cat.Get(fs.Arg(0))
	if err != nil {
		return err
	}
	cfg, _, err := loadConfig()
	if err != nil {
		return err
	}
	isDefault := (m.Task == registry.TaskEmbedding && cfg.Defaults.Embedding == m.ID) ||
		(m.Task == registry.TaskReranker && cfg.Defaults.Reranker == m.ID)
	if isDefault && !*force {
		return fmt.Errorf("%s es el default de %s; usá otro modelo o --force", m.ID, m.Task)
	}
	store, _, err := newStore()
	if err != nil {
		return err
	}
	if err := store.Remove(m.Task, m.ID); err != nil {
		return err
	}
	fmt.Printf("%s: eliminado\n", m.ID)
	return nil
}

func modelsVerify(args []string) error {
	fs := flag.NewFlagSet("models verify", flag.ContinueOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	store, _, err := newStore()
	if err != nil {
		return err
	}
	ids := fs.Args()
	if len(ids) == 0 {
		for _, m := range cat.Models {
			if _, ok := store.Status(m.Task, m.ID); ok {
				ids = append(ids, m.ID)
			}
		}
	}
	if len(ids) == 0 {
		fmt.Println("(nada instalado)")
		return nil
	}
	for _, id := range ids {
		m, err := cat.Get(id)
		if err != nil {
			return err
		}
		if err := store.Verify(m.Task, m.ID, m.Files); err != nil {
			fmt.Printf("%s: %v\n", id, err)
			continue
		}
		fmt.Printf("%s: ok\n", id)
	}
	return nil
}

// modelsSync downloads the files of the selected models, computes their
// sha256, and writes the pins to the user catalogue. It is a maintainer
// command: it is the only one that touches the network without installing.
func modelsSync(args []string) error {
	fs := flag.NewFlagSet("models sync", flag.ContinueOnError)
	all := fs.Bool("all", false, "todos los modelos del catálogo")
	fromStore := fs.Bool("from-store", false, "hashear los archivos ya instalados en vez de descargarlos")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	var targets []registry.Model
	if *all {
		targets = cat.Models
	} else {
		for _, id := range fs.Args() {
			m, err := cat.Get(id)
			if err != nil {
				return err
			}
			targets = append(targets, m)
		}
	}
	if len(targets) == 0 {
		return fmt.Errorf("%w: models sync <id>... | --all", errUsage)
	}

	out := registry.Catalog{Version: cat.Version, Updated: time.Now().Format("2006-01-02")}
	client := &http.Client{Timeout: 30 * time.Minute}
	store, _, err := newStore()
	if err != nil {
		return err
	}
	for _, m := range targets {
		for i := range m.Files {
			fmt.Printf("%s/%s: calculando sha256…\n", m.ID, m.Files[i].Dest)
			var sum string
			var size int64
			if *fromStore {
				dir, derr := store.Dir(m.Task, m.ID)
				if derr != nil {
					return derr
				}
				sum, size, err = hashFileAt(filepath.Join(dir, m.Files[i].Dest))
			} else {
				sum, size, err = hashURL(context.Background(), client, m.Files[i].URL)
			}
			if err != nil {
				return err
			}
			m.Files[i].SHA256 = sum
			if m.Files[i].SizeBytes == 0 {
				m.Files[i].SizeBytes = size
			}
		}
		out.Models = append(out.Models, m)
	}
	path, err := registry.UserCatalogPath()
	if err != nil {
		return err
	}
	data, err := registry.Marshal(out)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return err
	}
	fmt.Printf("pines escritos en %s\n", path)
	return nil
}

// hashFileAt returns the sha256 and size of a local file.
func hashFileAt(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// hashURL downloads a URL and returns its sha256 and size without storing it.
func hashURL(ctx context.Context, client *http.Client, url string) (string, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", 0, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("%s: %s", url, resp.Status)
	}
	h := sha256.New()
	n, err := io.Copy(h, resp.Body)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// --- doctor ----------------------------------------------------------------

func cmdDoctor(args []string) error {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, cfgPath, err := loadConfig()
	if err != nil {
		return err
	}
	fmt.Printf("config   %s\n", cfgPath)
	ok := true

	if err := cfg.Validate(); err != nil {
		fmt.Printf("config   ✗ %v\n", err)
		ok = false
	} else {
		fmt.Printf("config   ✓ válida\n")
	}

	if err := ensureWritable(cfg.Paths.Home); err != nil {
		fmt.Printf("home     ✗ %v\n", err)
		ok = false
	} else {
		fmt.Printf("home     ✓ %s\n", cfg.Paths.Home)
	}

	if st, err := os.Stat(cfg.Paths.Vault); err != nil {
		fmt.Printf("vault    ✗ %v\n", err)
		ok = false
	} else if !st.IsDir() {
		fmt.Printf("vault    ✗ no es un directorio\n")
		ok = false
	} else {
		fmt.Printf("vault    ✓ %s\n", cfg.Paths.Vault)
	}

	cat, err := openCatalog()
	if err != nil {
		fmt.Printf("catálogo ✗ %v\n", err)
		ok = false
		return finish(ok)
	}
	store, _, err := newStore()
	if err != nil {
		return err
	}
	for _, task := range []string{registry.TaskEmbedding, registry.TaskReranker} {
		m, err := cat.Default(task)
		if err != nil {
			fmt.Printf("%-8s ✗ %v\n", task, err)
			ok = false
			continue
		}
		if _, installed := store.Status(m.Task, m.ID); installed {
			fmt.Printf("%-8s ✓ %s instalado\n", task, m.ID)
		} else {
			fmt.Printf("%-8s ✗ %s no instalado (argosctl models install %s)\n", task, m.ID, m.ID)
			ok = false
		}
	}

	// Qdrant: reachable and consistent with the configured dimension.
	if host, portStr, err := net.SplitHostPort(cfg.Qdrant.URL); err == nil {
		if port, perr := strconv.Atoi(portStr); perr == nil {
			if st, serr := qstore.NewWithREST(host, port, cfg.Qdrant.RESTURL, cfg.Qdrant.SnapshotsPath); serr == nil {
				ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
				dim, derr := st.CollectionDim(ctx, cfg.Index.Collection)
				cancel()
				st.Close()
				switch {
				case derr != nil:
					// A missing collection is normal before the first index.
					fmt.Printf("qdrant   · %s: colección %s todavía no existe\n", cfg.Qdrant.URL, cfg.Index.Collection)
				case cfg.Index.Dims > 0 && dim != cfg.Index.Dims:
					fmt.Printf("qdrant   ✗ %s tiene %d dims y la config %d\n", cfg.Index.Collection, dim, cfg.Index.Dims)
					ok = false
				default:
					fmt.Printf("qdrant   ✓ %s · %s (%d dims, quant %s)\n", cfg.Qdrant.URL, cfg.Index.Collection, dim, orNone(cfg.Index.Quantization))
				}
			} else {
				fmt.Printf("qdrant   ✗ %v\n", serr)
				ok = false
			}
		}
	}

	// Docling is optional: it is reported, not required.
	if cfg.Layout.URL != "" && cfg.Layout.Mode != "never" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimSuffix(cfg.Layout.URL, "/")+"/health", nil)
		if resp, lerr := http.DefaultClient.Do(req); lerr == nil {
			resp.Body.Close()
			fmt.Printf("layout   ✓ %s (%s)\n", cfg.Layout.URL, cfg.Layout.Mode)
		} else {
			fmt.Printf("layout   · %s no responde (opcional; se usa pdftotext)\n", cfg.Layout.URL)
		}
		cancel()
	}

	gpus := resources.Detect()
	providers, perr := embed.AvailableProviders(ortLib(cfg))
	if perr != nil {
		fmt.Printf("runtime  ✗ %v\n", perr)
		ok = false
	} else {
		fmt.Printf("runtime  ✓ EP disponibles: %s\n", strings.Join(orNoneList(providers), ", "))
	}
	if len(gpus) == 0 {
		fmt.Printf("gpu      · ninguna detectada (CPU)\n")
	} else {
		for _, g := range gpus {
			kind := "integrada"
			if g.Dedicated() {
				kind = "dedicada"
			}
			name := ""
			if g.Name != "" {
				name = " (" + g.Name + ")"
			}
			avail := "no disponible en esta build"
			if containsString(providers, g.Provider()) {
				avail = "EP disponible"
			}
			if name != "" {
				fmt.Printf("gpu      ✓ card%d %s %s · driver %s · %s (%s)%s\n",
					g.Index, g.Vendor, kind, g.Driver, g.Provider(), avail, name)
			} else {
				fmt.Printf("gpu      ✓ card%d %s %s · driver %s · %s (%s)\n",
					g.Index, g.Vendor, kind, g.Driver, g.Provider(), avail)
			}
		}
		if best, ok := resources.Best(gpus); ok {
			fmt.Printf("device   %s → %s (embed %s, rerank %s)\n",
				best.Vendor, best.Provider(), cfg.Resources.Devices.Embed, cfg.Resources.Devices.Rerank)
		}
	}

	if path, err := exec.LookPath("docker"); err == nil {
		fmt.Printf("docker   ✓ %s\n", path)
	} else {
		fmt.Printf("docker   · no encontrado (normal dentro de un contenedor; los límites del host no aplican)\n")
	}
	return finish(ok)
}

func finish(ok bool) error {
	if !ok {
		return errors.New("doctor encontró problemas")
	}
	fmt.Println("\ntodo en orden")
	return nil
}

func ensureWritable(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".write-check-*")
	if err != nil {
		return err
	}
	name := f.Name()
	f.Close()
	return os.Remove(name)
}

// --- config ----------------------------------------------------------------

func cmdConfig(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("%w: config <path|list|get|set>", errUsage)
	}
	cfg, cfgPath, err := loadConfig()
	if err != nil {
		return err
	}
	switch args[0] {
	case "path":
		fmt.Println(cfgPath)
		return nil
	case "list":
		data, err := yaml.Marshal(cfg)
		if err != nil {
			return err
		}
		fmt.Print(string(data))
		return nil
	case "get":
		if len(args) != 2 {
			return fmt.Errorf("%w: config get <clave>", errUsage)
		}
		v, err := getKey(cfg, args[1])
		if err != nil {
			return err
		}
		fmt.Println(v)
		return nil
	case "set":
		if len(args) != 3 {
			return fmt.Errorf("%w: config set <clave> <valor>", errUsage)
		}
		if err := setKey(&cfg, args[1], args[2]); err != nil {
			return err
		}
		if err := cfg.Save(cfgPath); err != nil {
			return err
		}
		if args[1] == "index.dims" {
			warnDimsMismatch(cfg)
		}
		fmt.Printf("%s = %s\n", args[1], args[2])
		return nil
	case "reset":
		if len(args) != 2 {
			return fmt.Errorf("%w: config reset <clave>", errUsage)
		}
		if err := resetKey(cfgPath, args[1]); err != nil {
			return err
		}
		effective, _, err := loadConfig()
		if err == nil {
			if v, gerr := getKey(effective, args[1]); gerr == nil {
				fmt.Printf("%s = %s (default)\n", args[1], v)
			}
		}
		return nil
	default:
		return fmt.Errorf("%w: config %s", errUsage, args[0])
	}
}

func getKey(cfg config.Config, key string) (string, error) {
	switch key {
	case "paths.home":
		return cfg.Paths.Home, nil
	case "paths.vault":
		return cfg.Paths.Vault, nil
	case "paths.compose":
		return cfg.Paths.Compose, nil
	case "defaults.embedding":
		return cfg.Defaults.Embedding, nil
	case "defaults.reranker":
		return cfg.Defaults.Reranker, nil
	case "filters.include":
		return strings.Join(cfg.Filters.Include, ","), nil
	case "filters.exclude":
		return strings.Join(cfg.Filters.Exclude, ","), nil
	case "filters.exclude_dirs":
		return strings.Join(cfg.Filters.ExcludeDirs, ","), nil
	case "chunking.max_tokens":
		return fmt.Sprint(cfg.Chunking.MaxTokens), nil
	case "chunking.min_tokens":
		return fmt.Sprint(cfg.Chunking.MinTokens), nil
	case "chunking.overlap":
		return fmt.Sprint(cfg.Chunking.Overlap), nil
	case "resources.mode":
		return cfg.Resources.Mode, nil
	case "resources.devices.embed":
		return cfg.Resources.Devices.Embed, nil
	case "resources.devices.rerank":
		return cfg.Resources.Devices.Rerank, nil
	case "resources.threads":
		return fmt.Sprint(cfg.Resources.Threads), nil
	case "resources.batch":
		return fmt.Sprint(cfg.Resources.Batch), nil
	case "qdrant.url":
		return cfg.Qdrant.URL, nil
	case "qdrant.rest_url":
		return cfg.Qdrant.RESTURL, nil
	case "qdrant.snapshots_path":
		return cfg.Qdrant.SnapshotsPath, nil
	case "search.hybrid":
		return strconv.FormatBool(cfg.Search.Hybrid), nil
	case "service.url":
		return cfg.Service.URL, nil
	case "index.collection":
		return cfg.Index.Collection, nil
	case "index.dims":
		return fmt.Sprint(cfg.Index.Dims), nil
	case "index.quantization":
		return cfg.Index.Quantization, nil
	case "layout.url":
		return cfg.Layout.URL, nil
	case "layout.mode":
		return cfg.Layout.Mode, nil
	default:
		return "", fmt.Errorf("%w: clave %q", errUsage, key)
	}
}

func setKey(cfg *config.Config, key, value string) error {
	list := func() []string {
		var out []string
		for _, v := range strings.Split(value, ",") {
			if v = strings.TrimSpace(v); v != "" {
				out = append(out, v)
			}
		}
		return out
	}
	switch key {
	case "paths.home":
		cfg.Paths.Home = value
	case "paths.vault":
		cfg.Paths.Vault = value
	case "paths.compose":
		cfg.Paths.Compose = value
	case "defaults.embedding":
		cfg.Defaults.Embedding = value
	case "defaults.reranker":
		cfg.Defaults.Reranker = value
	case "filters.include":
		cfg.Filters.Include = list()
	case "filters.exclude":
		cfg.Filters.Exclude = list()
	case "filters.exclude_dirs":
		cfg.Filters.ExcludeDirs = list()
	case "chunking.max_tokens", "chunking.min_tokens", "chunking.overlap":
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("%s debe ser un número", key)
		}
		switch key {
		case "chunking.max_tokens":
			cfg.Chunking.MaxTokens = n
		case "chunking.min_tokens":
			cfg.Chunking.MinTokens = n
		default:
			cfg.Chunking.Overlap = n
		}
	case "service.url":
		cfg.Service.URL = value
	case "qdrant.url":
		cfg.Qdrant.URL = value
	case "qdrant.rest_url":
		cfg.Qdrant.RESTURL = strings.TrimSuffix(value, "/")
	case "qdrant.snapshots_path":
		cfg.Qdrant.SnapshotsPath = value
	case "search.hybrid":
		b, err := strconv.ParseBool(value)
		if err != nil {
			return fmt.Errorf("search.hybrid debe ser true o false")
		}
		cfg.Search.Hybrid = b
	case "index.collection":
		cfg.Index.Collection = value
	case "index.dims":
		if value == "auto" {
			cfg.Index.Dims = 0
			break
		}
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("index.dims debe ser un número o auto")
		}
		cfg.Index.Dims = n
	case "index.quantization":
		cfg.Index.Quantization = value
	case "layout.url":
		cfg.Layout.URL = value
	case "layout.mode":
		cfg.Layout.Mode = value
	case "resources.devices.embed":
		cfg.Resources.Devices.Embed = value
	case "resources.devices.rerank":
		cfg.Resources.Devices.Rerank = value
	case "resources.mode":
		cfg.Resources.Mode = value
	case "resources.threads", "resources.batch":
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("%s debe ser un número", key)
		}
		if key == "resources.threads" {
			cfg.Resources.Threads = n
		} else {
			cfg.Resources.Batch = n
		}
	default:
		return fmt.Errorf("%w: clave %q", errUsage, key)
	}
	return cfg.Validate()
}

// warnDimsMismatch is informational: index.dims only affects new collections,
// so changing it never rewrites an existing base.
func warnDimsMismatch(cfg config.Config) {
	mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
	list, err := mgr.List()
	if err != nil {
		return
	}
	for _, meta := range list {
		fmt.Fprintf(os.Stderr, "aviso: %s ya usa %d dims; index.dims solo aplica a bases nuevas (db reindex --new-model)\n",
			meta.Name, meta.Dims)
	}
}

// resetKey removes one key from the YAML file so the code default returns.
// It edits the raw document instead of saving the loaded struct, which would
// also materialize every other default.
func resetKey(path, key string) error {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("no hay configuración en %s", path)
	}
	if err != nil {
		return err
	}
	var doc map[string]any
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}
	parts := strings.Split(key, ".")
	cur := doc
	for _, part := range parts[:len(parts)-1] {
		next, ok := cur[part].(map[string]any)
		if !ok {
			return fmt.Errorf("clave %q no existe", key)
		}
		cur = next
	}
	last := parts[len(parts)-1]
	if _, ok := cur[last]; !ok {
		return fmt.Errorf("clave %q no existe", key)
	}
	delete(cur, last)
	out, err := yaml.Marshal(doc)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".config-*.yaml")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(out); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// --- index -----------------------------------------------------------------

func cmdIndex(args []string) error {
	fs := flag.NewFlagSet("index", flag.ContinueOnError)
	dry := fs.Bool("dry-run", false, "solo muestra qué archivos entrarían")
	full := fs.Bool("full", false, "ignora el manifiesto y re-embebe todo")
	withChunks := fs.Bool("chunks", false, "parsea y chunkea para contar (requiere el tokenizer del modelo)")
	maxTokens := fs.Int("max-tokens", 0, "tamaño máximo de chunk (default: config)")
	minTokens := fs.Int("min-tokens", 0, "fusiona chunks menores a esto (default: config)")
	overlap := fs.Int("overlap", -1, "solape en tokens (default: config)")
	limit := fs.Int("limit", 0, "procesa solo los primeros N archivos (para probar)")
	dbName := fs.String("db", "", "base a indexar (por defecto, la config)")
	pathFilter := fs.String("path", "", "indexa solo este archivo o carpeta")
	include := fs.String("include", "", "extensiones a incluir (coma): .md,.pdf")
	exclude := fs.String("exclude", "", "extensiones a excluir (coma): .png")
	excludeDirs := fs.String("exclude-dir", "", "directorios a saltear (coma)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, _, err := loadConfig()
	if err != nil {
		return err
	}
	if *dbName == "all" {
		mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
		list, err := mgr.List()
		if err != nil {
			return err
		}
		if len(list) == 0 {
			return fmt.Errorf("no hay bases registradas")
		}
		for _, meta := range list {
			fmt.Printf("\n== %s ==\n", meta.Name)
			if err := cmdIndex(rewriteDBArg(args, meta.Name)); err != nil {
				return fmt.Errorf("%s: %w", meta.Name, err)
			}
		}
		return nil
	}
	manifestPath := ""
	if *dbName != "" {
		mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
		meta, err := mgr.Get(*dbName)
		if err != nil {
			return err
		}
		cfg.Paths.Vault = meta.Source
		cfg.Index.Collection = meta.Collection
		cfg.Index.Dims = meta.Dims
		cfg.Index.Quantization = meta.Quantization
		cfg.Defaults.Embedding = meta.Model
		manifestPath = mgr.ManifestPath(meta.Name)
	}
	opts := vault.Options{
		Include:     cfg.Filters.Include,
		Exclude:     cfg.Filters.Exclude,
		ExcludeDirs: cfg.Filters.ExcludeDirs,
	}
	if *include != "" {
		opts.Include = splitList(*include)
	}
	if *exclude != "" {
		opts.Exclude = splitList(*exclude)
	}
	if *excludeDirs != "" {
		opts.ExcludeDirs = splitList(*excludeDirs)
	}

	files, err := vault.Walk(cfg.Paths.Vault, opts)
	if err != nil {
		return err
	}
	if *pathFilter != "" {
		files = filterByPath(files, *pathFilter)
		if len(files) == 0 {
			return fmt.Errorf("no hay archivos que coincidan con %s", *pathFilter)
		}
	}
	if *limit > 0 && *limit < len(files) {
		files = files[:*limit]
	}
	summary := vault.Summarize(files)

	fmt.Printf("vault    %s\n", cfg.Paths.Vault)
	fmt.Printf("incluye  %s\n", joinOr(opts.Include, "todo"))
	if len(opts.Exclude) > 0 {
		fmt.Printf("excluye  %s\n", strings.Join(opts.Exclude, ", "))
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "\nTIPO\tARCHIVOS\tTAMAÑO")
	exts := make([]string, 0, len(summary.ByExt))
	for ext := range summary.ByExt {
		exts = append(exts, ext)
	}
	sort.Strings(exts)
	for _, ext := range exts {
		st := summary.ByExt[ext]
		fmt.Fprintf(w, "%s\t%d\t%.1f MB\n", ext, st.Files, float64(st.Bytes)/(1<<20))
	}
	fmt.Fprintf(w, "total\t%d\t%.1f MB\n", summary.Total, float64(summary.Bytes)/(1<<20))
	w.Flush()

	chunkCfg := cfg.Chunking
	if *maxTokens > 0 {
		chunkCfg.MaxTokens = *maxTokens
	}
	if *minTokens > 0 {
		chunkCfg.MinTokens = *minTokens
	}
	if *overlap >= 0 {
		chunkCfg.Overlap = *overlap
	}

	if *withChunks {
		stats, err := chunkStats(cfg, files, chunkCfg)
		if err != nil {
			return err
		}
		fmt.Printf("\nchunks   %d (%.0f tokens promedio, %d tokens totales)\n",
			stats.Chunks, stats.AvgTokens, stats.Tokens)
		// Time scales with tokens, not chunks: 738 ms for 512 tokens after
		// pruning the KV outputs, and 128 ms for 512 with e5-base. Per-chunk
		// overhead is small and constant, so the token rate is the stable
		// estimate.
		fmt.Printf("estimado %.1f h con qwen3-embed-0.6b · %.1f h con e5-base (por tokens, medido)\n",
			float64(stats.Tokens)*1.44/3.6e6, float64(stats.Tokens)*0.25/3.6e6)
		if stats.Failed > 0 {
			fmt.Printf("archivos con error de parseo: %d\n", stats.Failed)
		}
	}

	if *dry {
		return nil
	}
	return runIndex(cfg, files, chunkCfg, *full, manifestPath)
}

// runIndex wires the real pipeline: model, Qdrant, manifest and ingest.
func runIndex(cfg config.Config, files []vault.File, chunkCfg config.Chunking, full bool, manifestPath string) error {
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	m, err := cat.Get(cfg.Defaults.Embedding)
	if err != nil {
		return err
	}
	modelStore := models.New(filepath.Join(cfg.Paths.Home, "models"))
	dir, err := modelStore.Dir(m.Task, m.ID)
	if err != nil {
		return err
	}
	if _, ok := modelStore.Status(m.Task, m.ID); !ok {
		return fmt.Errorf("%s no está instalado (argosctl models install %s)", m.ID, m.ID)
	}
	tk, err := tokenizers.FromFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("tokenizer de %s: %w", m.ID, err)
	}
	defer tk.Close()
	counter := func(text string) int {
		enc, err := tk.EncodeWithOptionsErr(text, true)
		if err != nil {
			return len(strings.Fields(text))
		}
		return len(enc.IDs)
	}

	warnTruncation(m, cfg.Index.Dims)
	emb, err := embed.NewORT(embed.ORTOptions{
		ModelPath:     filepath.Join(dir, "model.onnx"),
		TokenizerPath: filepath.Join(dir, "tokenizer.json"),
		LibraryPath:   ortLib(cfg),
		Name:          m.ID,
		Dims:          effectiveDims(m, cfg.Index.Dims),
		MaxTokens:     m.Specs.MaxTokens,
		Pooling:       m.Specs.Pooling,
		QueryInstruct: m.QueryInstruction,
		Prefixes:      m.Prefixes,
		Threads:       cfg.Resources.Threads,
		BatchSize:     cfg.Resources.Batch,
		Device:        cfg.Resources.Devices.Embed,
		GPUIndex:      -1,
	})
	if err != nil {
		return err
	}
	defer emb.Close()

	host, portStr, err := net.SplitHostPort(cfg.Qdrant.URL)
	if err != nil {
		return fmt.Errorf("qdrant.url debe ser host:puerto: %w", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return fmt.Errorf("qdrant.url puerto inválido: %w", err)
	}
	st, err := qstore.NewWithREST(host, port, cfg.Qdrant.RESTURL, cfg.Qdrant.SnapshotsPath)
	if err != nil {
		return err
	}
	defer st.Close()

	if err := os.MkdirAll(filepath.Join(cfg.Paths.Home, "data"), 0o755); err != nil {
		return err
	}
	if manifestPath == "" {
		manifestPath = filepath.Join(cfg.Paths.Home, "data", safeName(cfg.Index.Collection), "manifest.db")
	}
	man, err := manifest.Open(manifestPath)
	if err != nil {
		return err
	}
	defer man.Close()

	splitter := chunk.Splitter{
		Counter:   counter,
		MaxTokens: chunkCfg.MaxTokens,
		MinTokens: chunkCfg.MinTokens,
		Overlap:   chunkCfg.Overlap,
		Namespace: uuid.NameSpaceURL,
	}

	last := time.Now()
	stats, err := ingest.Run(
		context.Background(), files, parse.DefaultRegistry(), splitter, emb, st, man,
		ingest.Options{
			Collection:   cfg.Index.Collection,
			Model:        m.ID,
			Quantization: cfg.Index.Quantization,
			Full:         full,
			LayoutURL:    cfg.Layout.URL,
			LayoutMode:   cfg.Layout.Mode,
			BatchSize:    64,
			Progress: func(ev ingest.Event) {
				if time.Since(last) < 2*time.Second && ev.Current != ev.Total {
					return
				}
				fmt.Printf("\r[%d/%d] %d chunks · %d vectores nuevos · %d reusados",
					ev.Current, ev.Total, ev.Chunks, ev.Embed, ev.Reused)
				last = time.Now()
			},
		})
	if err != nil {
		return err
	}
	fmt.Printf("\nindexado: %d nuevos, %d actualizados, %d sin cambios, %d borrados\n",
		stats.Added, stats.Updated, stats.Skipped, stats.Removed)
	fmt.Printf("chunks %d · vectores nuevos %d · reusados %d · errores %d\n",
		stats.Chunks, stats.Embed, stats.Reused, len(stats.Errors))
	for i, e := range stats.Errors {
		if i == 5 {
			fmt.Println("  …")
			break
		}
		fmt.Println("  error:", e)
	}
	if len(stats.Errors) > 0 {
		return fmt.Errorf("%d archivos fallaron", len(stats.Errors))
	}
	return nil
}

// warnTruncation tells the user when a smaller vector is not Matryoshka: the
// model was not trained for it and quality drops faster.
func warnTruncation(m registry.Model, dims int) {
	if dims > 0 && dims < m.Specs.Dims && !m.Specs.MRL {
		fmt.Fprintf(os.Stderr, "aviso: %s no es MRL; truncar de %d a %d dims pierde más calidad de la esperada\n",
			m.ID, m.Specs.Dims, dims)
	}
}

// effectiveDims applies an explicit override, or the catalogue default for
// that model. Models without Matryoshka default to their native size, so a
// global 512 can no longer truncate e5 by accident.
func effectiveDims(m registry.Model, configured int) int {
	if configured > 0 {
		return configured
	}
	return m.EffectiveDims()
}

// safeName keeps a collection name usable as a directory.
func safeName(s string) string {
	clean := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			return r
		}
		return '_'
	}, s)
	if clean == "" {
		return "default"
	}
	return clean
}

// ortLib prefers the bundled runtime over the system package, which may be
// older than the ONNX API the binding needs.
func ortLib(cfg config.Config) string {
	lib := embed.DiscoverRuntime(cfg.Paths.Home)
	if lib == "/usr/lib/libonnxruntime.so" {
		return ""
	}
	return lib
}

type chunkSummary struct {
	Chunks    int
	Tokens    int
	AvgTokens float64
	Failed    int
}

// chunkStats parses and chunks the selected files with the real tokenizer of
// the default embedding model, so the numbers match what indexing would do.
func chunkStats(cfg config.Config, files []vault.File, chunkCfg config.Chunking) (chunkSummary, error) {
	cat, err := openCatalog()
	if err != nil {
		return chunkSummary{}, err
	}
	m, err := cat.Get(cfg.Defaults.Embedding)
	if err != nil {
		return chunkSummary{}, err
	}
	store := models.New(filepath.Join(cfg.Paths.Home, "models"))
	dir, err := store.Dir(m.Task, m.ID)
	if err != nil {
		return chunkSummary{}, err
	}
	tk, err := tokenizers.FromFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return chunkSummary{}, fmt.Errorf("tokenizer de %s: %w (¿models install?)", m.ID, err)
	}
	defer tk.Close()

	counter := func(text string) int {
		enc, err := tk.EncodeWithOptionsErr(text, true)
		if err != nil {
			return len(strings.Fields(text))
		}
		return len(enc.IDs)
	}
	splitter := chunk.Splitter{
		Counter:   counter,
		MaxTokens: chunkCfg.MaxTokens,
		MinTokens: chunkCfg.MinTokens,
		Overlap:   chunkCfg.Overlap,
		Namespace: uuid.NameSpaceURL,
	}
	reg := parse.DefaultRegistry()
	var out chunkSummary
	ctx := context.Background()
	for _, f := range files {
		if _, ok := reg.Lookup(f.Path); !ok {
			continue
		}
		doc, err := reg.Parse(ctx, f.Path, f.Rel)
		if err != nil {
			out.Failed++
			continue
		}
		chunks := splitter.Split(doc)
		out.Chunks += len(chunks)
		for _, c := range chunks {
			out.Tokens += c.Tokens
		}
	}
	if out.Chunks > 0 {
		out.AvgTokens = float64(out.Tokens) / float64(out.Chunks)
	}
	return out, nil
}

// filterByPath keeps the files under a given file or directory.
func filterByPath(files []vault.File, target string) []vault.File {
	abs, err := filepath.Abs(target)
	if err != nil {
		abs = target
	}
	var out []vault.File
	for _, f := range files {
		if f.Path == abs || strings.HasPrefix(f.Path, abs+string(filepath.Separator)) {
			out = append(out, f)
		}
	}
	return out
}

func splitList(v string) []string {
	var out []string
	for _, part := range strings.Split(v, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}

func joinOr(values []string, fallback string) string {
	if len(values) == 0 {
		return fallback
	}
	return strings.Join(values, ", ")
}

// --- db --------------------------------------------------------------------

func cmdDB(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("%w: db <create|list|info|rm>", errUsage)
	}
	cfg, _, err := loadConfig()
	if err != nil {
		return err
	}
	mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
	switch args[0] {
	case "create":
		return dbCreate(cfg, mgr, args[1:])
	case "list":
		return dbList(mgr)
	case "info":
		return dbInfo(mgr, args[1:])
	case "stats":
		return dbStats(mgr, args[1:])
	case "docs":
		return dbDocs(mgr, args[1:])
	case "reindex":
		return dbReindex(cfg, mgr, args[1:])
	case "rm":
		return dbRemove(cfg, mgr, args[1:])
	default:
		return fmt.Errorf("%w: db %s", errUsage, args[0])
	}
}

func dbCreate(cfg config.Config, mgr *dbpkg.Manager, args []string) error {
	// Flags are accepted before or after the path: `db create ~/x --name y`
	// is what people type, and the stdlib parser stops at the first argument.
	var name, model string
	var positional []string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--name", "-name":
			if i+1 < len(args) {
				i++
				name = args[i]
			}
		case "--model", "-model":
			if i+1 < len(args) {
				i++
				model = args[i]
			}
		default:
			positional = append(positional, args[i])
		}
	}
	if len(positional) != 1 {
		return fmt.Errorf("%w: db create <carpeta> [--name N] [--model M]", errUsage)
	}
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	modelID := model
	if modelID == "" {
		modelID = cfg.Defaults.Embedding
	}
	m, err := cat.Get(modelID)
	if err != nil {
		return err
	}
	dims := effectiveDims(m, cfg.Index.Dims)
	warnTruncation(m, dims)
	meta, err := mgr.Create(dbpkg.Meta{
		Name:         name,
		Source:       positional[0],
		Model:        m.ID,
		Dims:         dims,
		Quantization: cfg.Index.Quantization,
	})
	if err != nil {
		return err
	}

	host, portStr, err := net.SplitHostPort(cfg.Qdrant.URL)
	if err != nil {
		return fmt.Errorf("qdrant.url debe ser host:puerto: %w", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return err
	}
	st, err := qstore.NewWithREST(host, port, cfg.Qdrant.RESTURL, cfg.Qdrant.SnapshotsPath)
	if err != nil {
		return err
	}
	defer st.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := st.EnsureCollection(ctx, meta.Collection, meta.Dims, meta.Quantization); err != nil {
		_ = mgr.Remove(meta.Name)
		return err
	}
	if err := st.EnsurePayloadIndexes(ctx, meta.Collection); err != nil {
		_ = mgr.Remove(meta.Name)
		return err
	}

	fmt.Printf("base       %s\n", meta.Name)
	fmt.Printf("origen     %s\n", meta.Source)
	fmt.Printf("colección  %s (%d dims, quant %s)\n", meta.Collection, meta.Dims, orNone(meta.Quantization))
	fmt.Printf("modelo     %s\n", meta.Model)
	return nil
}

func dbList(mgr *dbpkg.Manager) error {
	list, err := mgr.List()
	if err != nil {
		return err
	}
	if len(list) == 0 {
		fmt.Println("(sin bases; creá una con `argosctl db create <carpeta>`)")
		return nil
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "NOMBRE\tMODELO\tDIMS\tCOLECCIÓN\tORIGEN")
	for _, m := range list {
		fmt.Fprintf(w, "%s\t%s\t%d\t%s\t%s\n", m.Name, m.Model, m.Dims, m.Collection, m.Source)
	}
	return w.Flush()
}

func dbInfo(mgr *dbpkg.Manager, args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("%w: db info <nombre>", errUsage)
	}
	meta, err := mgr.Get(args[0])
	if err != nil {
		return err
	}
	fmt.Printf("nombre     %s\n", meta.Name)
	fmt.Printf("origen     %s\n", meta.Source)
	fmt.Printf("colección  %s\n", meta.Collection)
	fmt.Printf("modelo     %s (%d dims, quant %s)\n", meta.Model, meta.Dims, orNone(meta.Quantization))
	fmt.Printf("creada     %s\n", meta.Created.Format(time.RFC3339))
	man, err := manifest.Open(mgr.ManifestPath(meta.Name))
	if err == nil {
		defer man.Close()
		if all, err := man.All(); err == nil {
			chunks := 0
			for _, f := range all {
				chunks += len(f.Chunks)
			}
			fmt.Printf("archivos   %d\n", len(all))
			fmt.Printf("chunks     %d\n", chunks)
		}
	}
	return nil
}

func dbRemove(cfg config.Config, mgr *dbpkg.Manager, args []string) error {
	var force bool
	var positional []string
	for _, a := range args {
		if a == "--force" || a == "-force" {
			force = true
			continue
		}
		positional = append(positional, a)
	}
	if len(positional) != 1 {
		return fmt.Errorf("%w: db rm <nombre> [--force]", errUsage)
	}
	meta, err := mgr.Get(positional[0])
	if err != nil {
		return err
	}
	if !force {
		return fmt.Errorf("esto borra la base %s y su colección; repetí con --force", meta.Name)
	}
	host, portStr, err := net.SplitHostPort(cfg.Qdrant.URL)
	if err == nil {
		if port, perr := strconv.Atoi(portStr); perr == nil {
			if st, serr := qstore.NewWithREST(host, port, cfg.Qdrant.RESTURL, cfg.Qdrant.SnapshotsPath); serr == nil {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				_ = st.DeleteCollection(ctx, meta.Collection)
				cancel()
				st.Close()
			}
		}
	}
	if err := mgr.Remove(meta.Name); err != nil {
		return err
	}
	fmt.Printf("%s: eliminada\n", meta.Name)
	return nil
}

func orNone(v string) string {
	if v == "" {
		return "none"
	}
	return v
}

// --- search ----------------------------------------------------------------

func cmdSearch(args []string) error {
	fs := flag.NewFlagSet("search", flag.ContinueOnError)
	k := fs.Int("k", 20, "pasajes a devolver")
	dbName := fs.String("db", "", "base a consultar")
	minScore := fs.Float64("min-score", 0, "descarta resultados por debajo")
	maxPerFile := fs.Int("max-per-file", 2, "máximo de pasajes por archivo")
	rerankMode := fs.String("rerank", "auto", "auto|on|off")
	mode := fs.String("mode", "", "dense|hybrid (default: search.hybrid)")
	rerankDepth := fs.Int("rerank-depth", 0, "candidatos que pasan por el reranker (0 = pool)")
	asJSON := fs.Bool("json", false, "salida JSON")
	explain := fs.Bool("explain", false, "incluye tokens y posición del chunk")
	filter := newKVFlag()
	fs.Var(filter, "filter", "filtro payload k=v (repetible)")
	// Accept the query before the flags: `search "x" --db y` is what people
	// type, and the stdlib parser stops at the first positional argument.
	withValue := map[string]bool{
		"-k": true, "--k": true, "--db": true, "--min-score": true,
		"--max-per-file": true, "--rerank": true, "--rerank-depth": true, "--mode": true,
	}
	var flagArgs, positional []string
	for i := 0; i < len(args); i++ {
		if withValue[args[i]] && i+1 < len(args) {
			flagArgs = append(flagArgs, args[i], args[i+1])
			i++
			continue
		}
		if strings.HasPrefix(args[i], "-") {
			flagArgs = append(flagArgs, args[i])
			continue
		}
		positional = append(positional, args[i])
	}
	if err := fs.Parse(flagArgs); err != nil {
		return err
	}
	if len(positional) != 1 {
		return fmt.Errorf("%w: search \"consulta\"", errUsage)
	}
	query := positional[0]
	if *mode != "" && *mode != "dense" && *mode != "hybrid" {
		return fmt.Errorf("%w: --mode %q no es dense|hybrid", errUsage, *mode)
	}
	cfg, _, err := loadConfig()
	if err != nil {
		return err
	}
	if *minScore > 0 && hybridEnabled(cfg, *mode) {
		fmt.Fprintln(os.Stderr, "aviso: en modo híbrido el score es RRF (~0.0x), no es comparable con coseno; --min-score necesita calibración")
	}
	// Fast path: the resident service already has the models loaded.
	if cfg.Service.URL != "" {
		if done, err := searchViaService(cfg, *dbName, query, *k, *maxPerFile, *minScore, *rerankMode, *mode, *rerankDepth, *asJSON, *explain, filter.Map()); done {
			return err
		}
	}
	if *dbName == "all" {
		return fmt.Errorf("search --db all requiere argosd corriendo (service.url)")
	}

	modelID := cfg.Defaults.Embedding
	collection := cfg.Index.Collection
	dims := cfg.Index.Dims
	if *dbName != "" && *dbName != "all" {
		mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
		meta, err := mgr.Get(*dbName)
		if err != nil {
			return err
		}
		modelID = meta.Model
		collection = meta.Collection
		dims = meta.Dims
	}

	cat, err := openCatalog()
	if err != nil {
		return err
	}
	m, err := cat.Get(modelID)
	if err != nil {
		return err
	}
	modelStore := models.New(filepath.Join(cfg.Paths.Home, "models"))
	dir, err := modelStore.Dir(m.Task, m.ID)
	if err != nil {
		return err
	}
	if _, ok := modelStore.Status(m.Task, m.ID); !ok {
		return fmt.Errorf("%s no está instalado (argosctl models install %s)", m.ID, m.ID)
	}
	emb, err := embed.NewORT(embed.ORTOptions{
		ModelPath:     filepath.Join(dir, "model.onnx"),
		TokenizerPath: filepath.Join(dir, "tokenizer.json"),
		LibraryPath:   ortLib(cfg),
		Name:          m.ID,
		Dims:          effectiveDims(m, dims),
		MaxTokens:     m.Specs.MaxTokens,
		Pooling:       m.Specs.Pooling,
		QueryInstruct: m.QueryInstruction,
		Prefixes:      m.Prefixes,
		Threads:       cfg.Resources.Threads,
		Device:        cfg.Resources.Devices.Embed,
		GPUIndex:      -1,
	})
	if err != nil {
		return err
	}
	defer emb.Close()

	host, portStr, err := net.SplitHostPort(cfg.Qdrant.URL)
	if err != nil {
		return err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return err
	}
	st, err := qstore.NewWithREST(host, port, cfg.Qdrant.RESTURL, cfg.Qdrant.SnapshotsPath)
	if err != nil {
		return err
	}
	defer st.Close()

	searcher := search.Searcher{
		Embedder:    emb,
		Store:       st,
		Collection:  collection,
		K:           *k,
		MaxPerFile:  *maxPerFile,
		MinScore:    *minScore,
		RerankDepth: *rerankDepth,
		Filter:      filter.Map(),
		Hybrid:      hybridEnabled(cfg, *mode),
	}

	useRerank := *rerankMode == "on"
	if *rerankMode == "auto" {
		useRerank = true // only if the model is installed; checked below
	}
	if useRerank {
		rm, err := cat.Get(cfg.Defaults.Reranker)
		if err != nil {
			return err
		}
		rdir, err := modelStore.Dir(rm.Task, rm.ID)
		if err != nil {
			return err
		}
		if _, ok := modelStore.Status(rm.Task, rm.ID); !ok {
			if *rerankMode == "on" {
				return fmt.Errorf("%s no está instalado (argosctl models install %s)", rm.ID, rm.ID)
			}
		} else {
			rr, err := rerank.New(rerank.Options{
				ModelPath:     filepath.Join(rdir, "model.onnx"),
				TokenizerPath: filepath.Join(rdir, "tokenizer.json"),
				LibraryPath:   ortLib(cfg),
				Device:        cfg.Resources.Devices.Rerank,
				GPUIndex:      -1,
				Threads:       cfg.Resources.Threads,
			})
			if err != nil {
				return err
			}
			defer rr.Close()
			searcher.Reranker = rr
		}
	}

	start := time.Now()
	results, err := searcher.Search(context.Background(), query)
	if err != nil {
		return err
	}
	elapsed := time.Since(start)

	if *asJSON {
		data, err := json.MarshalIndent(map[string]any{
			"query": query, "collection": collection,
			"elapsed_ms": elapsed.Milliseconds(), "hits": results,
		}, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(data))
		return nil
	}

	if len(results) == 0 {
		fmt.Println("(sin resultados)")
		return nil
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	if *explain {
		fmt.Fprintln(w, "SCORE\tTOKENS\tCHUNK\tARCHIVO\tSECCIÓN")
	} else {
		fmt.Fprintln(w, "SCORE\tARCHIVO\tSECCIÓN")
	}
	for _, r := range results {
		if *explain {
			fmt.Fprintf(w, "%.3f\t%d\t%d\t%s\t%s\n", r.Score, r.Tokens, r.ChunkIndex, r.Path, truncate(r.Heading, 70))
			continue
		}
		fmt.Fprintf(w, "%.3f\t%s\t%s\n", r.Score, r.Path, truncate(r.Heading, 70))
	}
	w.Flush()
	fmt.Printf("\n%d resultados en %d ms (%s)\n", len(results), elapsed.Milliseconds(), collection)
	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

// hybridEnabled resolves the per-command override against the config.
func hybridEnabled(cfg config.Config, mode string) bool {
	switch mode {
	case "dense":
		return false
	case "hybrid":
		return true
	default:
		return cfg.Search.Hybrid
	}
}

// searchViaService calls argosd. It returns done=false to fall back to the
// local pipeline when the service is not running.
func searchViaService(cfg config.Config, db, query string, k, maxPerFile int, minScore float64, rerank, mode string, depth int, asJSON, explain bool, filter map[string]string) (bool, error) {
	payload, err := json.Marshal(map[string]any{
		"query": query, "db": db, "k": k, "max_per_file": maxPerFile,
		"min_score": minScore, "rerank": rerank, "mode": mode, "rerank_depth": depth,
		"filter": filter,
	})
	if err != nil {
		return false, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimSuffix(cfg.Service.URL, "/")+"/v1/search", strings.NewReader(string(payload)))
	if err != nil {
		return false, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false, nil // service down: local fallback
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode != http.StatusOK {
		return true, fmt.Errorf("argosd: %s", strings.TrimSpace(string(raw)))
	}
	if asJSON {
		fmt.Println(strings.TrimSpace(string(raw)))
		return true, nil
	}
	var out struct {
		Collection string   `json:"collection"`
		Mode       string   `json:"mode"`
		ElapsedMS  int64    `json:"elapsed_ms"`
		Warnings   []string `json:"warnings"`
		Hits       []struct {
			DB         string  `json:"db"`
			Path       string  `json:"path"`
			Heading    string  `json:"heading"`
			Score      float64 `json:"score"`
			Tokens     int     `json:"tokens"`
			ChunkIndex int     `json:"chunk_index"`
		} `json:"hits"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return false, nil
	}
	for _, w := range out.Warnings {
		fmt.Fprintln(os.Stderr, "aviso:", w)
	}
	showDB := db == "all"
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	switch {
	case explain && showDB:
		fmt.Fprintln(w, "SCORE	DB	TOKENS	CHUNK	ARCHIVO	SECCIÓN")
	case explain:
		fmt.Fprintln(w, "SCORE	TOKENS	CHUNK	ARCHIVO	SECCIÓN")
	case showDB:
		fmt.Fprintln(w, "SCORE	DB	ARCHIVO	SECCIÓN")
	default:
		fmt.Fprintln(w, "SCORE	ARCHIVO	SECCIÓN")
	}
	for _, h := range out.Hits {
		switch {
		case explain && showDB:
			fmt.Fprintf(w, "%.3f	%s	%d	%d	%s	%s\n", h.Score, h.DB, h.Tokens, h.ChunkIndex, h.Path, truncate(h.Heading, 70))
		case explain:
			fmt.Fprintf(w, "%.3f	%d	%d	%s	%s\n", h.Score, h.Tokens, h.ChunkIndex, h.Path, truncate(h.Heading, 70))
		case showDB:
			fmt.Fprintf(w, "%.3f	%s	%s	%s\n", h.Score, h.DB, h.Path, truncate(h.Heading, 70))
		default:
			fmt.Fprintf(w, "%.3f	%s	%s\n", h.Score, h.Path, truncate(h.Heading, 70))
		}
	}
	w.Flush()
	fmt.Printf("\n%d resultados en %d ms (%s, %s, argosd)\n", len(out.Hits), out.ElapsedMS, out.Collection, out.Mode)
	return true, nil
}

// kvFlag collects repeated --filter k=v values.
type kvFlag map[string]string

func newKVFlag() kvFlag { return kvFlag{} }

func (f kvFlag) String() string { return "" }

func (f kvFlag) Set(v string) error {
	i := strings.IndexByte(v, '=')
	if i <= 0 {
		return fmt.Errorf("filtro inválido %q (usá k=v)", v)
	}
	f[v[:i]] = v[i+1:]
	return nil
}

func (f kvFlag) Map() map[string]string { return map[string]string(f) }

// --- service ---------------------------------------------------------------

func cmdService(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("%w: service <status|load|unload>", errUsage)
	}
	cfg, _, err := loadConfig()
	if err != nil {
		return err
	}
	if cfg.Service.URL == "" {
		return fmt.Errorf("service.url está vacío")
	}
	base := strings.TrimSuffix(cfg.Service.URL, "/")

	if args[0] == "status" {
		resp, err := http.Get(base + "/v1/status")
		if err != nil {
			return fmt.Errorf("argosd no responde en %s: %w", base, err)
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		var pretty any
		if json.Unmarshal(raw, &pretty) == nil {
			out, _ := json.MarshalIndent(pretty, "", "  ")
			fmt.Println(string(out))
			return nil
		}
		fmt.Println(string(raw))
		return nil
	}

	if args[0] != "load" && args[0] != "unload" {
		return fmt.Errorf("%w: service %s", errUsage, args[0])
	}
	kind := "all"
	if len(args) > 1 {
		kind = args[1]
	}
	body, _ := json.Marshal(map[string]string{"model": kind})
	resp, err := http.Post(base+"/v1/models/"+args[0], "application/json", strings.NewReader(string(body)))
	if err != nil {
		return fmt.Errorf("argosd no responde en %s: %w", base, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("argosd: %s", strings.TrimSpace(string(raw)))
	}
	fmt.Printf("%s %s: ok\n", args[0], kind)
	return nil
}

// --- snapshots -------------------------------------------------------------

func cmdSnapshot(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("%w: snapshot <create|list|delete|restore>", errUsage)
	}
	action := args[0]
	rest := args[1:]
	dbName := ""
	var name string
	for i := 0; i < len(rest); i++ {
		switch rest[i] {
		case "--db":
			if i+1 < len(rest) {
				i++
				dbName = rest[i]
			}
		default:
			name = rest[i]
		}
	}
	cfg, _, err := loadConfig()
	if err != nil {
		return err
	}
	collection := cfg.Index.Collection
	if dbName != "" {
		mgr := dbpkg.New(filepath.Join(cfg.Paths.Home, "dbs"))
		meta, err := mgr.Get(dbName)
		if err != nil {
			return err
		}
		collection = meta.Collection
	}
	host, portStr, err := net.SplitHostPort(cfg.Qdrant.URL)
	if err != nil {
		return err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return err
	}
	st, err := qstore.NewWithREST(host, port, cfg.Qdrant.RESTURL, cfg.Qdrant.SnapshotsPath)
	if err != nil {
		return err
	}
	defer st.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	switch action {
	case "create":
		snap, err := st.CreateSnapshot(ctx, collection)
		if err != nil {
			return err
		}
		fmt.Printf("%s\t%s\t%.1f MB\n", snap.Name, snap.CreatedAt, float64(snap.Size)/(1<<20))
		return nil
	case "list":
		snaps, err := st.ListSnapshots(ctx, collection)
		if err != nil {
			return err
		}
		if len(snaps) == 0 {
			fmt.Println("(sin snapshots)")
			return nil
		}
		w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
		fmt.Fprintln(w, "NOMBRE\tCREADO\tTAMAÑO")
		for _, snap := range snaps {
			fmt.Fprintf(w, "%s\t%s\t%.1f MB\n", snap.Name, snap.CreatedAt, float64(snap.Size)/(1<<20))
		}
		return w.Flush()
	case "delete":
		if name == "" {
			return fmt.Errorf("%w: snapshot delete <nombre>", errUsage)
		}
		if err := st.DeleteSnapshot(ctx, collection, name); err != nil {
			return err
		}
		fmt.Printf("%s: eliminado\n", name)
		return nil
	case "restore":
		if name == "" {
			return fmt.Errorf("%w: snapshot restore <nombre>", errUsage)
		}
		if err := st.RestoreSnapshot(ctx, collection, name); err != nil {
			return err
		}
		fmt.Printf("%s: restaurado en %s\n", name, collection)
		return nil
	default:
		return fmt.Errorf("%w: snapshot %s", errUsage, action)
	}
}

// memoryStates asks argosd which models are loaded. A down service is not an
// error here: the list is still useful offline.
func memoryStates(cfg config.Config) map[string]string {
	out := map[string]string{}
	if cfg.Service.URL == "" {
		return out
	}
	client := &http.Client{Timeout: 1500 * time.Millisecond}
	resp, err := client.Get(strings.TrimSuffix(cfg.Service.URL, "/") + "/v1/status")
	if err != nil {
		return out
	}
	defer resp.Body.Close()
	var st struct {
		Models []struct {
			Key   string `json:"key"`
			State string `json:"state"`
		} `json:"models"`
	}
	if json.NewDecoder(resp.Body).Decode(&st) != nil {
		return out
	}
	for _, m := range st.Models {
		id := m.Key
		if i := strings.IndexByte(id, ':'); i >= 0 {
			id = id[i+1:]
		}
		if i := strings.IndexByte(id, '/'); i >= 0 {
			id = id[:i]
		}
		out[id] = m.State
	}
	return out
}

func orNoneList(values []string) []string {
	if len(values) == 0 {
		return []string{"ninguno"}
	}
	return values
}

func containsString(list []string, v string) bool {
	for _, s := range list {
		if s == v {
			return true
		}
	}
	return false
}

// --- completion -------------------------------------------------------------

func cmdCompletion(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("%w: completion bash|zsh|fish", errUsage)
	}
	commands := "help init paths models config index search db service snapshot doctor version completion"
	switch args[0] {
	case "bash":
		fmt.Printf("_argosctl() { COMPREPLY=( $(compgen -W \"%s\" -- \"${COMP_WORDS[COMP_CWORD]}\") ); }\ncomplete -F _argosctl argosctl\n", commands)
	case "zsh":
		fmt.Printf(`#compdef argosctl
_arguments -C '1:comando:->cmd' '*::arg:->args'
case $state in
  cmd) _values 'comando' %s ;;
  args) _files ;;
esac
`, commands)
	case "fish":
		for _, c := range strings.Fields(commands) {
			fmt.Printf("complete -c argosctl -f -a %s\n", c)
		}
	default:
		return fmt.Errorf("%w: shell %q", errUsage, args[0])
	}
	return nil
}

// --- db stats / docs / reindex ---------------------------------------------

func dbStats(mgr *dbpkg.Manager, args []string) error {
	var name string
	asJSON := false
	for _, a := range args {
		if a == "--json" {
			asJSON = true
			continue
		}
		name = a
	}
	if name == "" {
		return fmt.Errorf("%w: db stats <nombre> [--json]", errUsage)
	}
	meta, err := mgr.Get(name)
	if err != nil {
		return err
	}
	man, err := manifest.Open(mgr.ManifestPath(meta.Name))
	if err != nil {
		return err
	}
	defer man.Close()
	all, err := man.All()
	if err != nil {
		return err
	}
	type stat struct{ files, chunks int }
	byExt := map[string]*stat{}
	for _, f := range all {
		ext := filepath.Ext(f.Rel)
		if ext == "" {
			ext = "(sin extensión)"
		}
		st := byExt[ext]
		if st == nil {
			st = &stat{}
			byExt[ext] = st
		}
		st.files++
		st.chunks += len(f.Chunks)
	}
	exts := make([]string, 0, len(byExt))
	for ext := range byExt {
		exts = append(exts, ext)
	}
	sort.Strings(exts)
	totalFiles, totalChunks := 0, 0
	for _, ext := range exts {
		st := byExt[ext]
		totalFiles += st.files
		totalChunks += st.chunks
	}
	if asJSON {
		byExtJSON := make(map[string]map[string]int, len(byExt))
		for ext, st := range byExt {
			byExtJSON[ext] = map[string]int{"files": st.files, "chunks": st.chunks}
		}
		out := map[string]any{
			"db":     meta.Name,
			"files":  totalFiles,
			"chunks": totalChunks,
			"by_ext": byExtJSON,
		}
		data, err := json.MarshalIndent(out, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(data))
		return nil
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "TIPO	ARCHIVOS	CHUNKS")
	for _, ext := range exts {
		st := byExt[ext]
		fmt.Fprintf(w, "%s	%d	%d\n", ext, st.files, st.chunks)
	}
	fmt.Fprintf(w, "total	%d	%d\n", totalFiles, totalChunks)
	return w.Flush()
}

func dbDocs(mgr *dbpkg.Manager, args []string) error {
	var name string
	offset, limit := 0, 50
	asJSON := false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--json":
			asJSON = true
		case "--offset":
			if i+1 < len(args) {
				i++
				offset, _ = strconv.Atoi(args[i])
			}
		case "--limit":
			if i+1 < len(args) {
				i++
				limit, _ = strconv.Atoi(args[i])
			}
		default:
			name = args[i]
		}
	}
	if name == "" {
		return fmt.Errorf("%w: db docs <nombre> [--offset N] [--limit N] [--json]", errUsage)
	}
	meta, err := mgr.Get(name)
	if err != nil {
		return err
	}
	man, err := manifest.Open(mgr.ManifestPath(meta.Name))
	if err != nil {
		return err
	}
	defer man.Close()
	all, err := man.All()
	if err != nil {
		return err
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Rel < all[j].Rel })
	if offset > len(all) {
		offset = len(all)
	}
	end := offset + limit
	if end > len(all) {
		end = len(all)
	}
	if asJSON {
		type doc struct {
			Path   string `json:"path"`
			Chunks int    `json:"chunks"`
			Hash   string `json:"hash"`
		}
		docs := make([]doc, 0, end-offset)
		for _, f := range all[offset:end] {
			docs = append(docs, doc{Path: f.Rel, Chunks: len(f.Chunks), Hash: f.Hash})
		}
		out, err := json.MarshalIndent(map[string]any{
			"db": meta.Name, "total": len(all), "offset": offset, "limit": limit, "documents": docs,
		}, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(out))
		return nil
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "ARCHIVO\tCHUNKS\tHASH")
	for _, f := range all[offset:end] {
		fmt.Fprintf(w, "%s\t%d\t%s\n", f.Rel, len(f.Chunks), firstN(f.Hash, 10))
	}
	w.Flush()
	fmt.Printf("\n%d de %d documentos\n", end-offset, len(all))
	return nil
}

func dbReindex(cfg config.Config, mgr *dbpkg.Manager, args []string) error {
	var name, newModel string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--new-model":
			if i+1 < len(args) {
				i++
				newModel = args[i]
			}
		default:
			name = args[i]
		}
	}
	if name == "" {
		return fmt.Errorf("%w: db reindex <nombre> [--new-model M]", errUsage)
	}
	meta, err := mgr.Get(name)
	if err != nil {
		return err
	}
	model := meta.Model
	if newModel != "" {
		model = newModel
	}
	cat, err := openCatalog()
	if err != nil {
		return err
	}
	m, err := cat.Get(model)
	if err != nil {
		return err
	}
	dims := meta.Dims
	if model != meta.Model {
		dims = effectiveDims(m, cfg.Index.Dims)
	}
	// New generation: a fresh base over the same source, so the old index
	// stays searchable until the new one finishes.
	baseName := name + "-v2"
	for i := 3; i < 100; i++ {
		if _, err := mgr.Get(baseName); err != nil {
			break
		}
		baseName = fmt.Sprintf("%s-v%d", name, i)
	}
	fresh, err := mgr.CreateGeneration(dbpkg.Meta{
		Name:         baseName,
		Source:       meta.Source,
		Collection:   "argos_" + baseName,
		Model:        model,
		Dims:         dims,
		Quantization: meta.Quantization,
	})
	if err != nil {
		return err
	}
	fmt.Printf("nueva generación: %s (%s, %d dims, colección %s)\n", fresh.Name, fresh.Model, fresh.Dims, fresh.Collection)
	fmt.Printf("siguiente paso: argosctl index --db %s\n", fresh.Name)
	return nil
}

func firstN(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// rewriteDBArg replaces the value of --db so a recursive call indexes one base.
func rewriteDBArg(args []string, name string) []string {
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--db" || a == "-db":
			out = append(out, a, name)
			i++
		case strings.HasPrefix(a, "--db=") || strings.HasPrefix(a, "-db="):
			out = append(out, a[:strings.IndexByte(a, '=')+1]+name)
		default:
			out = append(out, a)
		}
	}
	return out
}

// --- eval ------------------------------------------------------------------

type evalSet struct {
	Queries []struct {
		Query   string `yaml:"query"`
		Path    string `yaml:"path"`
		Heading string `yaml:"heading,omitempty"`
	} `yaml:"queries"`
}

// cmdEval runs the eval set through the resident service and reports recall@k
// and MRR. It needs argosd because loading the model per query would make the
// measurement meaningless.
func cmdEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	dbName := fs.String("db", "", "base a evaluar")
	k := fs.Int("k", 8, "profundidad de corte")
	file := fs.String("file", "testdata/eval.yaml", "set de consultas")
	rerank := fs.String("rerank", "off", "auto|on|off")
	mode := fs.String("mode", "", "dense|hybrid (default: search.hybrid)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *mode != "" && *mode != "dense" && *mode != "hybrid" {
		return fmt.Errorf("%w: --mode %q no es dense|hybrid", errUsage, *mode)
	}
	cfg, _, err := loadConfig()
	if err != nil {
		return err
	}
	if cfg.Service.URL == "" {
		return fmt.Errorf("eval requiere service.url (argosd)")
	}
	raw, err := os.ReadFile(*file)
	if err != nil {
		return err
	}
	var set evalSet
	if err := yaml.Unmarshal(raw, &set); err != nil {
		return fmt.Errorf("eval: %w", err)
	}
	if len(set.Queries) == 0 {
		return fmt.Errorf("eval: %s no tiene consultas", *file)
	}

	hits, found, mrr := 0, 0, 0.0
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "RESULTADO\tCONSULTA\tESPERADO")
	for _, q := range set.Queries {
		rank, err := rankOf(cfg, *dbName, q.Query, q.Path, *k, *rerank, *mode)
		if err != nil {
			return fmt.Errorf("eval %q: %w", q.Query, err)
		}
		hits++
		if rank > 0 {
			found++
			mrr += 1.0 / float64(rank)
			fmt.Fprintf(w, "ok (%d)\t%s\t%s\n", rank, truncate(q.Query, 60), q.Path)
			continue
		}
		fmt.Fprintf(w, "MISS\t%s\t%s\n", truncate(q.Query, 60), q.Path)
	}
	w.Flush()
	fmt.Printf("\nrecall@%d: %d/%d = %.1f%%\n", *k, found, hits, float64(found)/float64(hits)*100)
	fmt.Printf("MRR: %.3f\n", mrr/float64(hits))
	return nil
}

func rankOf(cfg config.Config, db, query, wantPath string, k int, rerank, mode string) (int, error) {
	body, _ := json.Marshal(map[string]any{"query": query, "db": db, "k": k, "rerank": rerank, "mode": mode, "max_per_file": 1})
	resp, err := http.Post(strings.TrimSuffix(cfg.Service.URL, "/")+"/v1/search", "application/json", strings.NewReader(string(body)))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return 0, fmt.Errorf("argosd: %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	var out struct {
		Hits []struct {
			Path string `json:"path"`
		} `json:"hits"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, err
	}
	for i, h := range out.Hits {
		if strings.TrimPrefix(h.Path, "./") == strings.TrimPrefix(wantPath, "./") {
			return i + 1, nil
		}
	}
	return 0, nil
}
