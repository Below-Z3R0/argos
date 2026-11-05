# Tools

Referencia de cada tool MCP. Todas ejecutan `argosctl`; el JSON de `search` es el mismo que produce
`argosctl search --json`.

## search

```json
{"query": "chunking markdown", "db": "knowledge", "k": 5, "rerank": "auto"}
```

| Campo | Tipo | Default | Notas |
|---|---|---|---|
| `query` | string | — | Obligatorio |
| `db` | string | config | Base a consultar (`all` incluido) |
| `model` | string | — | Filtra bases por modelo |
| `exclude_dbs` | array | — | Bases a excluir |
| `k` | int | 20 | Pasajes devueltos |
| `max_per_file` | int | 2 | Tope por archivo |
| `min_score` | float | 0 | Descarta por debajo |
| `rerank` | string | auto | `auto\|on\|off` |
| `mode` | string | config | `dense\|hybrid` |
| `rerank_depth` | int | pool | Candidatos que pasan por el reranker |

Devuelve `{query, collection, model, mode, elapsed_ms, hits:[{db, path, heading, text, score, tokens, chunk_index}], warnings?}`.
`warnings` aparece cuando el reranker estaba en `auto` y no se pudo cargar.

## db_list / db_info

`db_list` no recibe argumentos. `db_info` recibe `{name}` y suma archivos y chunks del manifiesto.

## db_create

```json
{"path": "/home/user/Documents/Knowledge", "name": "knowledge", "model": "qwen3-embed-0.6b"}
```

Crea el directorio de la base, su manifiesto y la colección en Qdrant. Requiere escritura.

## index

```json
{"db": "knowledge", "path": "docs/", "full": false, "dry_run": false}
```

- **Sin `dry_run`**: arranca un job y devuelve `{id, command, started}`. Se consulta con `job_status`.
- **Con `dry_run`**: ejecuta `argosctl index --dry-run` y devuelve el conteo; con `count_chunks` agrega
  el chunking real con el tokenizer.

`full` ignora el manifiesto y re-embebe; `path` limita a un archivo o carpeta.

## job_status / job_list / job_cancel

`job_status {id, offset?, limit?}` devuelve estado, un rango de la salida (`total_bytes`, `more`) y el
código de salida. `job_list` lista los últimos 20 jobs. `job_cancel {id}` corta un job en ejecución;
el job termina con `canceled: true`.

## eval

```json
{"db": "knowledge", "file": "testdata/eval.yaml", "k": 8, "mode": "hybrid"}
```

Arranca un job que mide recall@k y MRR contra `argosd`. Requiere escritura.

## models_catalog

`{task?}` filtra por `embedding` o `reranker`. Muestra tier, parámetros, dims, RAM, disco, licencia y
estado de instalación.

## models_use

`{id}` instala si falta y lo deja como default de su tarea. Requiere escritura.

## config_get / config_set

`config_get {key}` y `config_set {key, value}`. Claves permitidas: cualquier prefijo de
`paths.`, `defaults.`, `filters.`, `chunking.`, `resources.`, `qdrant.`, `index.`, `layout.`.
El CLI valida el valor (por ejemplo `index.quantization` solo acepta `none|int8`).

## doctor

Sin argumentos. Verifica config, rutas, modelos, Qdrant, GPUs, Docling opcional y Docker. Sale con
error solo ante problemas reales.

## paths

Sin argumentos. Muestra `home`, `models`, `data`, `qdrant`, `vault` y `compose`.

## Errores

Un `argosctl` con código distinto de cero devuelve `isError: true` y el texto de salida. Los errores de
argumentos se resuelven en el CLI, no en el servidor MCP.
