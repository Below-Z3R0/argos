// Package tools exposes the Argos CLI as MCP tools.
//
// Every tool shells out to argosctl: the CLI already validates input, resolves
// config and formats errors, and duplicating that in a second binary would
// drift. The MCP layer only adds: JSON contracts, async jobs and a read-only
// switch.
package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/Below-Z3R0/argos/internal/mcp/daemon"
	"github.com/Below-Z3R0/argos/internal/mcp/jobs"
	"github.com/Below-Z3R0/argos/internal/mcp/runner"
	"github.com/Below-Z3R0/argos/internal/metrics"
)

// Deps are the collaborators the tools need.
type Deps struct {
	Runner   *runner.Runner
	Jobs     *jobs.Manager
	Daemon   *daemon.Client
	Metrics  *metrics.Registry
	ReadOnly bool
}

// Register adds every tool to the server.
func Register(s *mcp.Server, d Deps) {
	mcp.AddTool(s, &mcp.Tool{
		Name:        "search",
		Description: "Busca pasajes en una base de Argos. Devuelve JSON con path, heading, text y score.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in SearchInput) (*mcp.CallToolResult, any, error) {
		// Prefer the resident service: it keeps the models loaded. The CLI
		// fallback works, but pays a model load on every call.
		if d.Daemon != nil {
			raw, err := d.Daemon.Search(ctx, map[string]any{
				"query": in.Query, "db": in.DB, "k": in.K,
				"model": in.Model, "exclude_dbs": in.ExcludeDBs,
				"max_per_file": in.MaxPerFile, "min_score": in.MinScore,
				"rerank": in.Rerank, "mode": in.Mode, "rerank_depth": in.RerankDepth,
				"filter": in.Filter,
			})
			if err == nil {
				d.inc(metrics.Labels{"source": "daemon"})
				return text(string(raw)), nil, nil
			}
		}
		args := []string{"search", in.Query, "--json", "--local"}
		if in.DB != "" {
			args = append(args, "--db", in.DB)
		}
		if in.Model != "" {
			args = append(args, "--model", in.Model)
		}
		if len(in.ExcludeDBs) > 0 {
			args = append(args, "--exclude-db", strings.Join(in.ExcludeDBs, ","))
		}
		if in.K > 0 {
			args = append(args, "--k", fmt.Sprint(in.K))
		}
		if in.MaxPerFile > 0 {
			args = append(args, "--max-per-file", fmt.Sprint(in.MaxPerFile))
		}
		if in.MinScore > 0 {
			args = append(args, "--min-score", fmt.Sprint(in.MinScore))
		}
		if in.Rerank != "" {
			args = append(args, "--rerank", in.Rerank)
		}
		if in.Mode != "" {
			args = append(args, "--mode", in.Mode)
		}
		if in.RerankDepth > 0 {
			args = append(args, "--rerank-depth", fmt.Sprint(in.RerankDepth))
		}
		for k, v := range in.Filter {
			args = append(args, "--filter", k+"="+v)
		}
		if in.PathPrefix != "" {
			args = append(args, "--path-prefix", in.PathPrefix)
		}
		return d.run(ctx, args...)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "service_status",
		Description: "Estado del servicio argosd: colección, modelo y modelos cargados en memoria.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		if d.Daemon == nil {
			return d.fail(errors.New("argosd no está configurado (--argosd http://127.0.0.1:7712)")), nil, nil
		}
		raw, err := d.Daemon.Status(ctx)
		if err != nil {
			return d.fail(err), nil, nil
		}
		return text(string(raw)), nil, nil
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "db_list",
		Description: "Lista las bases de Argos con su modelo, dimensión y carpeta de origen.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		return d.run(ctx, "db", "list")
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "db_info",
		Description: "Muestra el detalle de una base: origen, colección, modelo, archivos y chunks.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in DBNameInput) (*mcp.CallToolResult, any, error) {
		return d.run(ctx, "db", "info", in.Name)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "db_create",
		Description: "Crea una base apuntando a una carpeta. Escribe en disco y en Qdrant.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in DBCreateInput) (*mcp.CallToolResult, any, error) {
		if err := d.writable(); err != nil {
			return d.fail(err), nil, nil
		}
		args := []string{"db", "create", in.Path}
		if in.Name != "" {
			args = append(args, "--name", in.Name)
		}
		if in.Model != "" {
			args = append(args, "--model", in.Model)
		}
		return d.run(ctx, args...)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "db_rm",
		Description: "Borra una base y su colección. Requiere force=true y escritura habilitada.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in DBRmInput) (*mcp.CallToolResult, any, error) {
		if err := d.writable(); err != nil {
			return d.fail(err), nil, nil
		}
		if !in.Force {
			return d.fail(errors.New("borrar requiere force=true")), nil, nil
		}
		return d.run(ctx, "db", "rm", in.Name, "--force")
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "models_load",
		Description: "Carga en memoria un modelo por id, o embed/rerank/all para el default.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in ModelKindInput) (*mcp.CallToolResult, any, error) {
		if d.Daemon == nil {
			return d.fail(errors.New("argosd no está configurado")), nil, nil
		}
		raw, err := d.Daemon.Load(ctx, in.Model)
		if err != nil {
			return d.fail(err), nil, nil
		}
		return text(string(raw)), nil, nil
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "models_unload",
		Description: "Libera de memoria un modelo por id, o embed/rerank/all. La próxima búsqueda lo recarga.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in ModelKindInput) (*mcp.CallToolResult, any, error) {
		if d.Daemon == nil {
			return d.fail(errors.New("argosd no está configurado")), nil, nil
		}
		raw, err := d.Daemon.Unload(ctx, in.Model)
		if err != nil {
			return d.fail(err), nil, nil
		}
		return text(string(raw)), nil, nil
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "models_install",
		Description: "Descarga e instala un modelo del catálogo, sin cambiar el default.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in ModelInput) (*mcp.CallToolResult, any, error) {
		if err := d.writable(); err != nil {
			return d.fail(err), nil, nil
		}
		return d.run(ctx, "models", "install", "--yes", in.ID)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "index",
		Description: "Indexa una base. Sin dry_run arranca un job en segundo plano y devuelve job_id; con dry_run cuenta y sale.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in IndexInput) (*mcp.CallToolResult, any, error) {
		if !in.DryRun {
			if err := d.writable(); err != nil {
				return d.fail(err), nil, nil
			}
		}
		args := []string{"index"}
		if in.DB != "" {
			args = append(args, "--db", in.DB)
		}
		if in.Path != "" {
			args = append(args, "--path", in.Path)
		}
		if in.Full {
			args = append(args, "--full")
		}
		if in.DryRun {
			args = append(args, "--dry-run")
			if in.CountChunks {
				args = append(args, "--chunks")
			}
			return d.run(ctx, args...)
		}
		job, err := d.Jobs.Start(ctx, args...)
		if err != nil {
			return d.fail(err), nil, nil
		}
		out, _ := json.MarshalIndent(job, "", "  ")
		return text(string(out)), nil, nil
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "job_status",
		Description: "Consulta un job de indexación por id: estado, salida y código de salida.",
	}, func(_ context.Context, _ *mcp.CallToolRequest, in JobInput) (*mcp.CallToolResult, any, error) {
		job, ok := d.Jobs.GetWindow(in.ID, in.Offset, in.Limit)
		if !ok {
			return d.fail(fmt.Errorf("job %s no existe", in.ID)), nil, nil
		}
		out, _ := json.MarshalIndent(job, "", "  ")
		return text(string(out)), nil, nil
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "job_list",
		Description: "Lista los jobs recientes de esta sesión del servidor MCP.",
	}, func(_ context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		out, _ := json.MarshalIndent(d.Jobs.List(), "", "  ")
		return text(string(out)), nil, nil
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "job_cancel",
		Description: "Cancela un job en ejecución. Devuelve si el job existía y seguía corriendo.",
	}, func(_ context.Context, _ *mcp.CallToolRequest, in JobInput) (*mcp.CallToolResult, any, error) {
		if !d.Jobs.Cancel(in.ID) {
			return d.fail(fmt.Errorf("job %s no existe o ya terminó", in.ID)), nil, nil
		}
		return text(fmt.Sprintf("job %s cancelado", in.ID)), nil, nil
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "eval",
		Description: "Mide recall@k y MRR sobre un set de consultas. Corre como job; requiere argosd.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in EvalInput) (*mcp.CallToolResult, any, error) {
		if err := d.writable(); err != nil {
			return d.fail(err), nil, nil
		}
		args := []string{"eval"}
		if in.File != "" {
			args = append(args, "--file", in.File)
		}
		if in.DB != "" {
			args = append(args, "--db", in.DB)
		}
		if in.K > 0 {
			args = append(args, "--k", fmt.Sprint(in.K))
		}
		if in.Rerank != "" {
			args = append(args, "--rerank", in.Rerank)
		}
		if in.Mode != "" {
			args = append(args, "--mode", in.Mode)
		}
		job, err := d.Jobs.Start(ctx, args...)
		if err != nil {
			return d.fail(err), nil, nil
		}
		out, _ := json.MarshalIndent(job, "", "  ")
		return text(string(out)), nil, nil
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "models_catalog",
		Description: "Lista el catálogo de modelos con specs, licencia y estado de instalación.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in CatalogInput) (*mcp.CallToolResult, any, error) {
		args := []string{"models", "catalog"}
		if in.Task != "" {
			args = append(args, "--task", in.Task)
		}
		return d.run(ctx, args...)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "models_use",
		Description: "Instala si falta y deja un modelo como default de su tarea. Escribe configuración.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in ModelInput) (*mcp.CallToolResult, any, error) {
		if err := d.writable(); err != nil {
			return d.fail(err), nil, nil
		}
		return d.run(ctx, "models", "use", in.ID)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "config_get",
		Description: "Lee una clave de configuración (paths.home, index.dims, resources.devices.embed…).",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in ConfigInput) (*mcp.CallToolResult, any, error) {
		return d.run(ctx, "config", "get", in.Key)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "config_set",
		Description: "Escribe una clave de configuración. Solo claves conocidas; el CLI valida el valor.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in ConfigSetInput) (*mcp.CallToolResult, any, error) {
		if err := d.writable(); err != nil {
			return d.fail(err), nil, nil
		}
		if !allowedKey(in.Key) {
			return d.fail(fmt.Errorf("clave no permitida: %s", in.Key)), nil, nil
		}
		return d.run(ctx, "config", "set", in.Key, in.Value)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "snapshot_create",
		Description: "Crea un snapshot consistente de una base (por defecto la de la config).",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in DBOptInput) (*mcp.CallToolResult, any, error) {
		if err := d.writable(); err != nil {
			return d.fail(err), nil, nil
		}
		return d.run(ctx, snapshotArgs("create", in.DB, "")...)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "snapshot_list",
		Description: "Lista los snapshots de una base con fecha y tamaño.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in DBOptInput) (*mcp.CallToolResult, any, error) {
		return d.run(ctx, snapshotArgs("list", in.DB, "")...)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "snapshot_delete",
		Description: "Borra un snapshot por nombre.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in SnapshotInput) (*mcp.CallToolResult, any, error) {
		if err := d.writable(); err != nil {
			return d.fail(err), nil, nil
		}
		return d.run(ctx, snapshotArgs("delete", in.DB, in.Name)...)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "snapshot_restore",
		Description: "Restaura una base desde un snapshot. Sobrescribe la colección.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in SnapshotInput) (*mcp.CallToolResult, any, error) {
		if err := d.writable(); err != nil {
			return d.fail(err), nil, nil
		}
		return d.run(ctx, snapshotArgs("restore", in.DB, in.Name)...)
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "doctor",
		Description: "Verifica entorno: config, rutas, modelos, Qdrant, GPU y Docker.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		return d.run(ctx, "doctor")
	})

	mcp.AddTool(s, &mcp.Tool{
		Name:        "paths",
		Description: "Muestra las rutas efectivas: home, modelos, data, vault y compose.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		return d.run(ctx, "paths")
	})
}

// --- inputs ----------------------------------------------------------------

type SearchInput struct {
	Query       string            `json:"query" jsonschema:"consulta en lenguaje natural"`
	DB          string            `json:"db,omitempty" jsonschema:"base a consultar; vacío usa la colección de la config"`
	Model       string            `json:"model,omitempty" jsonschema:"filtra bases por modelo"`
	ExcludeDBs  []string          `json:"exclude_dbs,omitempty" jsonschema:"bases a excluir"`
	K           int               `json:"k,omitempty" jsonschema:"pasajes a devolver (default 20)"`
	MaxPerFile  int               `json:"max_per_file,omitempty" jsonschema:"máximo de pasajes por archivo"`
	MinScore    float64           `json:"min_score,omitempty" jsonschema:"descarta resultados por debajo de este score"`
	Rerank      string            `json:"rerank,omitempty" jsonschema:"auto|on|off"`
	Mode        string            `json:"mode,omitempty" jsonschema:"dense|hybrid; vacío usa la config"`
	RerankDepth int               `json:"rerank_depth,omitempty" jsonschema:"candidatos que pasan por el reranker"`
	Filter      map[string]string `json:"filter,omitempty" jsonschema:"filtros de payload, por ejemplo {\"lang\":\"es\"}"`
	PathPrefix  string            `json:"path_prefix,omitempty" jsonschema:"limita a una carpeta del corpus, por ejemplo work o learning/projects"`
}

type DBNameInput struct {
	Name string `json:"name" jsonschema:"nombre de la base"`
}

// EvalInput drives the recall harness. A file only makes sense inside the
// container, where testdata is mounted with the repo.
type EvalInput struct {
	DB     string `json:"db,omitempty" jsonschema:"base a evaluar"`
	File   string `json:"file,omitempty" jsonschema:"set YAML de consultas (default testdata/eval.yaml)"`
	K      int    `json:"k,omitempty" jsonschema:"profundidad de corte (default 8)"`
	Rerank string `json:"rerank,omitempty" jsonschema:"auto|on|off"`
	Mode   string `json:"mode,omitempty" jsonschema:"dense|hybrid; vacío usa la config"`
}

type DBCreateInput struct {
	Path  string `json:"path" jsonschema:"carpeta raíz a indexar"`
	Name  string `json:"name,omitempty" jsonschema:"nombre de la base; por defecto el de la carpeta"`
	Model string `json:"model,omitempty" jsonschema:"modelo de embeddings"`
}

type IndexInput struct {
	DB          string `json:"db,omitempty" jsonschema:"base a indexar"`
	Path        string `json:"path,omitempty" jsonschema:"archivo o carpeta puntual"`
	Full        bool   `json:"full,omitempty" jsonschema:"ignora el manifiesto y re-embebe todo"`
	DryRun      bool   `json:"dry_run,omitempty" jsonschema:"solo cuenta, no escribe"`
	CountChunks bool   `json:"count_chunks,omitempty" jsonschema:"en dry_run, parsea y cuenta chunks con el tokenizer"`
}

type DBOptInput struct {
	DB string `json:"db,omitempty" jsonschema:"base; vacío usa la colección de la config"`
}

type SnapshotInput struct {
	DB   string `json:"db,omitempty" jsonschema:"base"`
	Name string `json:"name" jsonschema:"nombre del snapshot"`
}

type DBRmInput struct {
	Name  string `json:"name" jsonschema:"nombre de la base"`
	Force bool   `json:"force" jsonschema:"confirmación explícita"`
}

type JobInput struct {
	ID     string `json:"id" jsonschema:"id del job"`
	Offset int    `json:"offset,omitempty" jsonschema:"offset en bytes para paginar la salida"`
	Limit  int    `json:"limit,omitempty" jsonschema:"bytes a devolver (default 16 KiB)"`
}

type CatalogInput struct {
	Task string `json:"task,omitempty" jsonschema:"embedding|reranker"`
}

type ModelKindInput struct {
	Model string `json:"model" jsonschema:"embed|rerank|all o id del modelo"`
}

type ModelInput struct {
	ID string `json:"id" jsonschema:"id del modelo"`
}

type ConfigInput struct {
	Key string `json:"key" jsonschema:"clave punteada"`
}

type ConfigSetInput struct {
	Key   string `json:"key" jsonschema:"clave punteada"`
	Value string `json:"value" jsonschema:"valor"`
}

// --- helpers ---------------------------------------------------------------

func (d Deps) run(ctx context.Context, args ...string) (*mcp.CallToolResult, any, error) {
	res, err := d.Runner.Run(ctx, args...)
	if err != nil {
		d.failCount()
		return d.fail(err), nil, nil
	}
	if res.ExitCode != 0 {
		d.failCount()
	} else {
		d.okCount()
	}
	out := res.Stdout
	if res.Stderr != "" {
		if out != "" {
			out += "\n"
		}
		out += res.Stderr
	}
	if res.ExitCode != 0 {
		msg := strings.TrimSpace(out)
		if msg == "" {
			msg = fmt.Sprintf("argosctl salió con código %d", res.ExitCode)
		}
		return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: msg}}}, nil, nil
	}
	return text(out), nil, nil
}

func (d Deps) okCount() {
	if d.Metrics != nil {
		d.Metrics.Inc("argos_mcp_calls_total", metrics.Labels{"outcome": "ok"})
	}
}

func (d Deps) failCount() {
	if d.Metrics != nil {
		d.Metrics.Inc("argos_mcp_calls_total", metrics.Labels{"outcome": "error"})
	}
}

func (d Deps) inc(labels metrics.Labels) {
	if d.Metrics != nil {
		d.Metrics.Inc("argos_mcp_search_total", labels)
	}
}

func (d Deps) fail(err error) *mcp.CallToolResult {
	d.failCount()
	return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: err.Error()}}}
}

func (d Deps) writable() error {
	if d.ReadOnly {
		return fmt.Errorf("el servidor está en modo solo lectura")
	}
	return nil
}

func text(s string) *mcp.CallToolResult {
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: s}}}
}

// allowedKey keeps the tool from writing arbitrary YAML keys.
func allowedKey(key string) bool {
	prefixes := []string{
		"paths.", "defaults.", "filters.", "chunking.", "resources.",
		"qdrant.", "index.", "layout.",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(key, p) {
			return true
		}
	}
	return false
}

func snapshotArgs(action, db, name string) []string {
	args := []string{"snapshot", action}
	if name != "" {
		args = append(args, name)
	}
	if db != "" {
		args = append(args, "--db", db)
	}
	return args
}
