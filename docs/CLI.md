# CLI de Argos — referencia y cobertura

Dos binarios: `argosctl` (operación) y `argosd` (servicio, sin subcomandos).

## Convenciones globales

| Flag | Efecto |
|---|---|
| `--config PATH` | Config alternativo (default `~/.config/argos/config.yaml`) |
| `--data-dir PATH` | Bases y modelos (default `~/.local/share/argos`) |
| `--json` | Salida máquina-legible en cualquier comando |
| `--progress auto\|plain\|json` | Barra en TTY, líneas JSON si no hay TTY |
| `--quiet` / `-q` | Solo errores |
| `-v` / `--verbose` | Log de debug a stderr |
| `--no-color` | Sin ANSI |
| `--timeout DUR` | Corta la operación (default: sin límite en index, 30 s en red) |

Exit codes: `0` ok · `1` uso inválido · `2` entorno (Qdrant, modelo, disco) · `3` interrumpido · `4`
datos inválidos (ruta inexistente, base corrupta).

## Comandos

### Catálogo y modelos

```
argosctl models list            [--all] [--task embedding|reranker] [--json]
argosctl models catalog         [--task embedding|reranker] [--tier T] [--sort ram|disk|name] [--json]
argosctl models show <id>       [--json]
argosctl models use [<id>]      [--task T] [--yes]   # sin id: elige de la lista
argosctl models install <id>... [--yes] [--allow-unpinned]
argosctl models remove <id>     [--force]
argosctl models verify [<id>...]
argosctl models sync            # mantenedor: resuelve URLs y fija sha256
argosctl models update          # trae el catálogo remoto
argosctl models path
```

- `list`: solo instalados, una línea por modelo.
- `catalog`: **el catálogo completo con especificaciones**, para elegir. Incluye tier, params, dims,
  RAM, disco, licencia, estado (instalado / default / no instalado) y el link oficial.
- `show`: ficha completa de uno, incluida la lista de archivos y su hash.
- `use`: instala si falta y lo deja por defecto (flujo en [`MODELS.md`](MODELS.md)).

### Bases

```
argosctl db create <ruta> [--name N] [--model M]
argosctl db list
argosctl db info <N>
argosctl db rm <N> --force
```

`db create` crea `$ARGOS_HOME/dbs/<nombre>/` con `meta.json` y la colección `argos_<nombre>`.
Si el nombre existe, agrega un sufijo aleatorio y reporta el nombre final. Con `index --db N` se usan
el origen, el modelo y las dims de esa base.

`info` responde lo que el plugin necesita para la ficha de la base: ruta origen, modelo, dims,
generación, archivos, puntos, tamaño en disco, última corrida y errores.

### Indexación

```
argosctl index [--db NOMBRE] [--path P] [--full] [--dry-run] [--chunks]
               [--include .md,.pdf] [--exclude .png] [--exclude-dir .git]
               [--device auto|cpu|gpu] [--threads N] [--ram-limit MB]
               [--progress ...] [--json]
```

- Incremental por defecto (hash + mtime); `--full` ignora el manifiesto.
- `--path` indexa un archivo o directorio puntual con el mismo diff.
- `--dry-run` informa altas, modificaciones y borrados sin tocar Qdrant.
- Los overrides de `--device/--threads/--ram-limit` valen solo para esta corrida.

### Búsqueda

```
argosctl search "<q>" [--db NOMBRE] [-k 20] [--max-per-file 2]
                      [--rerank auto|on|off] [--rerank-depth N]
                      [--min-score F] [--json]
```

- `--db` elige la base; sin él usa la colección de la config.
- `--rerank auto` usa el reranker si está instalado; `on` falla si falta; `off` solo vectorial.
- `--rerank-depth 0` rerankea todo el pool (`k × 3`).
- `--json` devuelve `{query, collection, elapsed_ms, hits}`.
- La búsqueda es híbrida (dense + sparse + RRF). Los scores son de fusión (~0.016), así que
  `--min-score` todavía no está calibrado.
- `eval --file testdata/eval.yaml --db N` mide recall@k y MRR contra el set.
- `models list` separa `INSTALADO` (disco), `DEFAULT` y `MEMORIA` (consultado a `argosd`).
- Si `service.url` responde, `search` va por `argosd` (modelos residentes, ~50 ms); si no, cae al
  pipeline local (carga el modelo, ~3 s).

Las bases son tablas independientes; se puede buscar en una o en varias.

| Invocación | Alcance |
|---|---|
| `search "x"` | La base default (`search.default_db`) |
| `search "x" --db learning` | Una sola tabla |
| `search "x" --db learning,university` | Varias tablas explícitas |
| `search "x" --db all` | Todas las bases |
| `search "x" --db all --model qwen3-embed-0.6b` | Todas las que usan ese modelo |
| `search "x" --db all --exclude-db borradores` | Todas menos una |

Reglas:

- El conjunto se resuelve por intersección: `--db` primero, después `--model` filtra, después
  `--exclude-db` resta.
- `--model` acepta el id de una entrada del catálogo o `all` (sin filtro).
- Si el conjunto queda vacío, exit code 4 con el detalle de qué filtro lo vació.
- La fusión es **por rango** (RRF), así que funciona aunque las bases usen modelos distintos: los
  scores no se comparan, solo las posiciones.
- `--explain` muestra base, sección, rango por rama (dense/sparse) y posición final.

### Recursos y entorno

```
argosctl gpu list                 [--json]
argosctl resources show           [--json]      # límites del contenedor + consumo
argosctl resources set            [--cpus N] [--memory 2g] [--pids N] [--device /dev/dri]
                                  [--mode auto|cpu|gpu|mixed] [--threads N]
                                  [--gpu ID|auto]
argosctl doctor                   [--json]
argosctl service status
argosctl service load  [embed|rerank|all]
argosctl service unload [embed|rerank|all]
argosctl snapshot create|list [--db N]
argosctl snapshot delete|restore <nombre> [--db N]
argosctl init                     # crea ${ARGOS_HOME} y el .env del stack
argosctl paths                    [--json]
```

- `resources set` escribe la config (no hace falta editar YAML a mano).
- `doctor` verifica Qdrant, colecciones, modelo default instalado y con hash correcto, threads
  efectivos, RAM disponible y GPUs.
- `status` consulta el servicio si está corriendo (`/v1/status`) y, si no, muestra el estado local.

### Servicio

```
argosctl serve [--addr 127.0.0.1:7700] [--mcp]
```

### Utilidades

```
argosctl version
argosctl completion bash|zsh|fish
argosctl config path|list|get <k>|set <k> <v>
```

## Matriz de cobertura

| Necesidad | Comando | Estado |
|---|---|---|
| Crear base apuntando a una carpeta | `db create <ruta>` | cubierto |
| Base por carpeta con colección propia | `db create` + `index --db` | **agregado** |
| Nombre por defecto = carpeta, colisión con sufijo | `db create` | cubierto |
| Elegir modelo al crear | `db create --model` | cubierto |
| Ver todas las bases | `db list` | cubierto |
| Ver tamaño y contenido de una base | `db info`, `db stats` | implementado |
| Listar documentos/chunks (plugin) | `db docs` | implementado |
| Borrar base | `db rm` | cubierto |
| Reindexar todo o parte | `index --full`, `index --path` | cubierto |
| Solo archivos nuevos | diff automático | cubierto |
| Cambiar de modelo | `models use` + `db reindex --new-model` | cubierto |
| Ver catálogo con specs para elegir | `models catalog` | **agregado** |
| Instalar / quitar modelo | `models install/remove` | cubierto |
| Buscar en una base (single table) | `search --db N` | cubierto |
| Buscar en varias tablas | `search --db a,b` | cubierto |
| Buscar en todas | `search --db all` | implementado |
| Buscar solo las bases de un modelo | `search --model ID` | **agregado** |
| Excluir una base de la búsqueda | `--exclude-db` | **agregado** |
| Reordenar por relevancia real | `--rerank auto` (default) | **agregado** |
| Cambiar el reranker sin tocar el embedder | `models use <id-reranker>` (task sale del catálogo) o `models use --task reranker` | **agregado** |
| Usar otro reranker solo en una consulta | `search --rerank-model ID` | **agregado** |
| Ajustar costo/latencia del rerank | `--rerank-depth` | **agregado** |
| Elegir GPU/CPU por corrida | `index --device` | **agregado** |
| Limitar hilos y RAM | `index --threads/--ram-limit`, `resources set` | **agregado** |
| Ver GPUs detectadas | `gpu list` | cubierto |
| Ver progreso y porcentaje | `--progress`, SSE | cubierto |
| Ver estado del servicio | `service status` | **agregado** |
| Mover el estado a otro disco | `config set paths.home` + `init` | **agregado** |
| Ver/crear las rutas del stack | `paths`, `init` | **agregado** |
| Limitar CPU/RAM del contenedor | `resources set --cpus/--memory` | **agregado** |
| Editar config sin YAML | `config set`, `resources set` | **agregado** |
| Autocompletado en fish | `completion fish` | implementado |
| Ver qué archivos entrarían | `index --dry-run` | **agregado** |
| Contar chunks con el tokenizer real | `index --dry-run --chunks` | **agregado** |
| Incluir/excluir tipos de archivo | `--include/--exclude` + `config set filters.*` | **agregado** |
| Versión para soporte | `version` | **agregado** |

## Huecos que quedaron fuera (a propósito)

| Cosa | Por qué no ahora |
|---|---|
| `db rename` | Renombrar implica mover colección y manifiesto; se hace cuando haya demanda real |
| `index --watch` | `fsnotify` agrega un modo de ejecución permanente; está en el roadmap |
| `db export/import` | El índice es derivado: se reconstruye, no se exporta |
| `search --rerank` | Depende del reranker, que es una fase posterior |
| Autenticación | El servicio vive en loopback; se agrega si sale de ahí |
