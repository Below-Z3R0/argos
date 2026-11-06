# CLI de Argos — referencia

Dos binarios: `argosctl` (operación) y `argosd` (servicio). `argos-mcp` expone el CLI por MCP.

No hay flags globales (`--config`, `--json`, `--quiet`, etc.) en esta versión: cada subcomando declara
sus propios flags. La configuración se resuelve por variables de entorno y
`$ARGOS_HOME/config.yaml` (por defecto `~/Services/argos/config.yaml`).

## Variables de entorno

| Variable | Efecto |
|---|---|
| `ARGOS_HOME` | Raíz de estado y config (`config.yaml`, `models.yaml`, `data/`, `models/`, `dbs/`, `runtime/`). Por defecto `~/Services/argos`. |
| `ARGOS_CONFIG` | Ruta explícita del `config.yaml`. Gana sobre `ARGOS_HOME`. |
| `ARGOS_COMPOSE_DIR` | Checkout del repo donde `init` escribe el `.env`. Por defecto `~/Documents/Github/argos`. |
| `ARGOS_VAULT` | Corpus por defecto. |
| `ARGOS_QDRANT_URL` | Endpoint gRPC (`host:port`). |
| `ARGOS_SERVICE_URL` | URL de `argosd` para el fast path del CLI. |
| `ARGOS_ORT_LIB` | `libonnxruntime.so` a usar. |
| `ARGOS_LAYOUT_URL` / `ARGOS_LAYOUT` | Docling y modo `auto|always|never`. |

## init / paths

```text
argosctl init [--home DIR] [--vault DIR] [--no-compose]
argosctl paths
```

`init` crea las carpetas de estado, guarda la config y escribe el `.env` del compose cuando
`paths.compose` existe.

## Modelos

```text
argosctl models list     [--task embedding|reranker] [--all]
argosctl models catalog  [--task embedding|reranker] [--tier TIER]
argosctl models show     <id>
argosctl models install  <id>... [--yes] [--allow-unpinned]
argosctl models use      <id> [--yes] [--allow-unpinned]
argosctl models remove   <id> [--force]
argosctl models verify   [<id>...]
argosctl models sync     [--all] [--from-store] [<id>...]
```

- `list`: modelos instalados y estado de descarga; `--all` incluye los que faltan.
- `catalog`: catálogo completo con specs, licencia y estado.
- `show`: ficha de un modelo, archivos y hash.
- `install`: descarga y verifica SHA-256; no cambia el default.
- `use`: instala si falta y deja el modelo como default de su tarea. Escribe la config; el servicio
  lo aplica al reiniciar o con `service load <modelo>`.
- `remove`: rechaza modelos usados por una base o cargados en `argosd`, salvo `--force`.
- `verify`: recalcula los hashes de los archivos instalados.
- `sync`: comando de mantenedor; resuelve URLs, calcula SHA-256 y escribe el catálogo de usuario.

No existen `models update` ni `models path`.

## Config

```text
argosctl config path
argosctl config list
argosctl config get <clave>
argosctl config set <clave> <valor>
argosctl config reset <clave>
```

Claves soportadas: `paths.*`, `defaults.embedding`, `defaults.reranker`, `filters.*`,
`chunking.*`, `qdrant.url`, `qdrant.rest_url`, `qdrant.snapshots_path`, `index.collection`,
`index.dims`, `index.quantization`, `search.hybrid`, `layout.url`, `layout.mode`,
`resources.mode`, `resources.devices.embed`, `resources.devices.rerank`, `resources.threads`,
`resources.batch`.

`config set index.dims auto` equivale a `config reset index.dims`: usa el default del catálogo.
`reset` borra la clave del YAML crudo; no materializa el resto de los defaults.

## Indexación

```text
argosctl index [--db NOMBRE] [--path P] [--full] [--dry-run] [--chunks]
               [--include .md,.pdf] [--exclude .png] [--exclude-dir .git]
               [--limit N] [--max-tokens N] [--min-tokens N] [--overlap N]
```

- Incremental por defecto: si `mtime`, tamaño y modelo no cambiaron, no lee el archivo.
- `--full` ignora el manifiesto y re-embebe; borra los puntos viejos después del upsert.
- `--dry-run` informa el plan sin tocar Qdrant.
- `--chunks` cuenta chunks con el tokenizer real del modelo.
- `--db all` recorre todas las bases registradas.
- No existen `--device`, `--threads` ni `--ram-limit`; el device y los threads salen de la config.

## Bases

```text
argosctl db create <ruta> [--name N] [--model M]
argosctl db list
argosctl db info <N>
argosctl db stats <N> [--json]
argosctl db docs <N> [--offset N] [--limit N] [--json]
argosctl db reindex <N> [--new-model M]
argosctl db rm <N> --force
```

- `create`: el modelo debe estar instalado. Crea `meta.json` y la colección.
- `info`: metadata + archivos/chunks del manifiesto. No consulta puntos reales en Qdrant.
- `stats` / `docs`: leen el manifiesto; `--json` para consumo de máquina.
- `reindex`: crea una generación nueva (`-v2`) y deja la vieja consultable; después hay que indexar
  la nueva y retirar la vieja a mano.
- `rm`: borra la colección y la metadata.

## Búsqueda

```text
argosctl search "<q>" [--db NOMBRE|a,b|all] [--model M] [--exclude-db N]
                      [-k 20] [--max-per-file 2]
                      [--rerank auto|on|off] [--rerank-depth N]
                      [--mode dense|hybrid] [--min-score F]
                      [--filter k=v ...] [--json] [--explain] [--local]
```

- `--db` elige base, lista o `all`; sin él usa `index.collection`.
- `--local` no consulta `argosd`.
- `--rerank auto` usa el reranker si está instalado; `on` falla si falta; `off` solo vectorial.
- `--mode` vacío usa `search.hybrid`.
- En híbrido el score es RRF (~0.0x), no coseno; `--min-score` no está calibrado.
- `--json` devuelve el JSON del engine; `--explain` agrega tokens y chunk a la tabla.
- No existe `--rerank-model`.

## Servicio

```text
argosctl service status
argosctl service load  [embed|rerank|all|<modelo>]
argosctl service unload [embed|rerank|all|<modelo>]
```

`service load <modelo>` precarga todas las dimensiones que usan las bases registradas para ese
modelo. `service unload <modelo>` libera solo esas sesiones. `embed|rerank|all` operan por tarea.

`argosd` se levanta aparte:

```text
argosd --addr 127.0.0.1:7712 [--preload none|embed|rerank|all|<modelo>]
```

## Snapshots

```text
argosctl snapshot create  [--db N]
argosctl snapshot list    [--db N]
argosctl snapshot delete  <nombre> [--db N]
argosctl snapshot restore <nombre> [--db N]
```

## Eval

```text
argosctl eval [--db N] [--k 8] [--file testdata/eval.yaml] [--rerank auto|on|off] [--mode dense|hybrid]
```

Requiere `argosd`; mide recall@k y MRR contra el set de consultas.

## Doctor

```text
argosctl doctor
```

Verifica config, rutas, catálogo, modelos default instalados, Qdrant, Docling opcional, EPs de ORT,
GPUs y Docker. No tiene `--json`. No verifica hashes de modelos, RAM disponible ni threads efectivos.

## Utilidades

```text
argosctl version
argosctl completion bash|zsh|fish
```

## Exit codes

| Código | Significado |
|---|---|
| `0` | OK |
| `1` | Uso inválido |
| `2` | Error de entorno, modelo, datos o Qdrant |

Los códigos `3` (interrumpido), `4` (datos inválidos) y `5` (job falló) **no están implementados**
en esta versión.
