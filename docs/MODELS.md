# Modelos y catálogo (Argos)

## ¿Existe una tienda de modelos?

No hay un estándar que sirva para este caso. Lo que existe:

| Opción | Qué da | Por qué no alcanza |
|---|---|---|
| Hugging Face Hub API | Metadatos, archivos, descargas | No dice cuánta RAM necesita ni si corre en CPU; hay decenas de exportaciones ONNX del mismo modelo con calidades distintas |
| Registro de Ollama | Modelos listos para `ollama run` | Atado a su runtime; no sirve para ONNX directo |
| Catálogos de LM Studio / Jan | Modelos con tamaño y cuantización | Atados a su app, sin API estable ni specs de embeddings |
| `llmfit` y similares | Recomendación por hardware | Recomendador, no catálogo mantenible ni instalable |

Conclusión: **catálogo propio**, versionado en el repo, curado por nosotros. Es la única forma de
tener specs comparables (params, dims, RAM, cuantización), tiers y un link oficial por modelo.

## Formato

`internal/registry/catalog.yaml`, embebido con `go:embed`. El usuario puede sobreescribir entradas
en `/home/b3low-z3r0/Services/argos/models.yaml` (mismo esquema, gana la última).

```yaml
version: 1
updated: "2026-11-05"
models:
  - id: e5-base                    # estable; es lo que se guarda en la base
    name: multilingual-e5-base     # para mostrar
    task: embedding
    tier: balanced                # balanced | fast | quality | compat | light | max | minimal
    default: true
    license: Apache-2.0
    homepage: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    specs:
      params: "0.6B"
      dims: 1024
      default_dims: 512          # tamaño que usa Argos si index.dims = 0
      max_tokens: 32768
      pooling: last_token
      quant: int8
      ram_mb: 900
      disk_mb: 596
    query_instruction: "..."      # se aplica en el embedder, nunca en el llamador
    files:
      - url: https://...
        dest: model.onnx
        size_mb: 585
        sha256: ""                # lo completa `models sync`
```

Reglas del esquema:

- `id` no cambia nunca: es lo que queda grabado en `meta.json` de cada base.
- `sha256` vacío significa **no instalable** hasta que se complete (`--allow-unpinned` lo permite,
  con advertencia).
- `specs.default_dims` es el tamaño efectivo cuando `index.dims` no se setea. Para modelos MRL es la
  truncación probada (Qwen3: 512 de 1024); para modelos sin MRL es el tamaño nativo (e5: 768, bge-m3:
  1024). `config set index.dims N` lo overrides; `config reset index.dims` vuelve al catálogo.
- `specs.ram_mb` es un **presupuesto conservador** (pesos + arena de ORT + activaciones del lote), no
  una medición. El spike de la fase 0 mide el RSS real y reemplaza el valor.
- `homepage` apunta al modelo oficial; `files[].url` al artefacto exacto.

### Orden de listado

Se ordena por tier, y dentro del tier por RAM ascendente:

1. `balanced` (el default)
2. `fast`
3. `quality`
4. `compat` (respaldos probados)
5. `light` (desarrollo)
6. `max` (solo con hardware)
7. `minimal`

El default se marca con `*`. La razón del orden: primero lo que hay que elegir, después lo que se
elige por una razón concreta (velocidad, respaldo, hardware limitado).

## Estado instalado

Hay **dos stores de modelos**, separados por tarea:

```
~/Services/argos/models/
├── embeddings/<id>/     {model.onnx, tokenizer.json, installed.json}
└── rerankers/<id>/      {model.onnx, tokenizer.json, installed.json}
```

Se separan porque se cambian por separado: podés tener el embedder en CPU y el reranker en GPU, o
cambiar uno sin tocar el otro. Cada `installed.json` lleva `task`, y los defaults son independientes
(`defaults.embedding` y `defaults.reranker`).

Presencia = directorio + `installed.json` válido + hashes correctos. Un directorio a medias se
considera no instalado y se limpia.

## CLI

```
argosctl models list [--all] [--json]     # compacto; --all incluye no instalados
argosctl models catalog [--tier T] [--json]   # catálogo completo con especificaciones
argosctl models show <id>
argosctl models use <id> [--task T] [--yes]          # instala si falta, después lo pone por defecto
argosctl models install <id>... [--yes]
argosctl models remove <id> [--force]
argosctl models verify [<id>...]          # recalcula sha256
argosctl models sync                      # mantenedor: descarga, calcula hashes y fija pines
argosctl models sync --from-store <id>    # fija pines de lo ya instalado, sin volver a bajar
argosctl models update                    # trae el catálogo remoto (opcional)
argosctl models path
```

### Rerankers

El catálogo tiene dos `task`: `embedding` y `reranker`. `models catalog --task reranker` muestra solo
los segundos, y el default de cada task es independiente.

| id | Params | RAM | Disco | Licencia |
|---|---|---|---|---|
| `bge-reranker-v2-m3-int8` * | 568M | 800 MB | 560 MB | MIT |
| `qwen3-reranker-0.6b` | 0.6B | 1.6 GB | 1.2 GB | Apache-2.0 |

El default es **BGE v2-M3**: MIT, multilingüe, ~71.5 BEIR y 560 MB. Se descartó jina v2 (mejor peso) por
su licencia CC-BY-NC, que bloquea cualquier uso comercial. `qwen3-reranker-0.6b` es la alternativa
Apache: export seq-cls, un logit y sigmoid, sin los 151k logits del CausalLM original. Pide 1.2 GB,
así que no es default en una máquina de 8 GB.

**Elegir el reranker por CLI** — tres formas, ninguna ambigua:

```sh
argosctl models use bge-reranker-v2-m3-int8     # la tarea sale del catálogo (task: reranker)
argosctl models use --task reranker           # sin id: muestra los rerankers y deja elegir
argosctl models catalog --task reranker       # ver todos antes de elegir
argosctl models list --task reranker          # ver los instalados
```

Y para una consulta puntual, sin cambiar el default:

```sh
argosctl search "qdrant" --rerank on --rerank-depth 60
```

El `id` identifica la tarea porque el catálogo tiene `task` por entrada; `--task` existe para filtrar
listas y para el picker, no para desambiguar ids.

### `models catalog`: la vista para elegir

Es el comando que responde "¿cuál me conviene?". Muestra todo el catálogo con las specs que
importan, en el orden de tiers definido arriba:

```
TIER       ID                    PARAMS  DIMS  RAM      DISCO   LICENCIA    ESTADO
balanced*  e5-base               278M     768   800 MB   282 MB  MIT         instalado · default
fast       embeddinggemma-300m   300M     768   380 MB   208 MB  Gemma       no instalado
quality    qwen3-embed-0.6b      0.6B     1024  1.6 GB   596 MB  Apache-2.0  instalado
compat     bge-m3                568M     1024  750 MB   559 MB  MIT         no instalado
```

La RAM está medida con chunks de 512 tokens, no con frases sueltas: el pico de un modelo con KV
cache crece con la longitud de entrada.

Flags: `--tier` filtra por familia de uso, `--sort ram|disk|name`, `--json` para el plugin.
La ficha completa (URLs, hashes, notas, homepage) sigue estando en `models show <id>`.

### Flujo de `models use <id>`

1. Busca el `id` en el catálogo (si no existe, sugiere los más parecidos).
2. Si el `sha256` está vacío → error: "modelo sin pin; corré `models sync`".
3. Verifica el estado instalado; si está sano, salta al paso 7.
4. Muestra: nombre, params, dims, RAM estimada, tamaño de descarga, licencia y `homepage`.
   Sin `--yes`, pide confirmación.
5. Comprueba espacio libre en disco (necesita `disk_mb × 1.2`).
6. Descarga cada archivo a `<id>/.tmp`, verifica sha256 y renombra atómicamente. Si falla, borra el
   `.tmp` y deja el estado anterior intacto. El progreso se muestra en stdout; no hay flag
   `--progress` en esta versión.
7. Escribe `default` en `~/Services/argos/config.yaml`.
8. Avisa si hay bases creadas con otro modelo (no se tocan).

### `models remove`

Se niega si:

- es el default (sin `--force`),
- alguna base lo tiene fijado en `meta.json` (sin `--force`),
- está cargado en `argosd` (sin `--force`).

Con `--force` borra el directorio y avisa qué bases quedan apuntando a un modelo ausente.

## Integración con el resto del CLI

- `db create` sin `--model` usa el default del catálogo; con `--model` fija ese.
- Cada base guarda `model`, `dims` y `quantization` en `meta.json`. Cambiar el default **no** las
  afecta, porque los vectores de otro modelo no son comparables.
- Las dims de una base nueva salen de `specs.default_dims` del modelo; `index.dims` solo las override
  y `config reset index.dims` vuelve al catálogo.
- `db reindex --new-model M` crea una generación nueva con ese modelo.
- `search --db all` agrupa las bases por modelo, calcula el vector de la consulta una vez por modelo
  distinto y fusiona por **rango** (RRF): los scores de modelos distintos no se comparan.
- `doctor` reporta modelo default, presencia y hash.

## Runtime

El modelo no alcanza: el vector depende del runtime.

- **onnxruntime 1.31** pinneado (host y Docker). Con un modelo int8, otra versión cambia los valores:
  un índice generado con 1.31 y consultado con otra build devuelve vectores de otro espacio.
- **libtokenizers 1.27** (`libtokenizers.a`) para parsear el `tokenizer.json` de Qwen3.
- `argosctl doctor` reporta la versión del runtime además de la del modelo.

## Estados

Hay dos ejes independientes y no se mezclan:

| Eje | Valores | Dónde vive |
|---|---|---|
| **Artefacto** (disco) | `missing`, `downloading` (%), `ready`, `failed` | `installed.json` + `.state.json` |
| **Runtime** (memoria) | `not_loaded`, `loading`, `ready`, `failed` | `/v1/status` de `argosd` |

`models list` los muestra en columnas separadas:

```
TAREA  ID  RAM  DISCO  INSTALADO  DEFAULT  MEMORIA
```

Antes había una sola columna "ESTADO" que juntaba "instalado", "default" y "descargando"; era
ambiguo. Si `argosd` no está corriendo, `MEMORIA` queda en `—`.

## Seguridad

- Solo `https`, y solo hosts permitidos (`huggingface.co`, `github.com`, `raw.githubusercontent.com`).
- El catálogo nunca ejecuta nada: son archivos de datos.
- `sha256` obligatorio salvo `--allow-unpinned` explícito.
- Antes de descargar se imprime la URL completa; el usuario ve de dónde baja.
- El catálogo remoto (`models update`) se valida contra el esquema y no puede cambiar `id`s ya
  usados por una base.

## Mantenimiento del catálogo

- **Agregar un modelo**: se agrega la entrada con `sha256` vacío, se corre `models sync` (descarga,
  hashea, reescribe) y se prueba con el eval set de 30 consultas antes de marcarlo `default`.
- **Deprecar**: se quita `default`, se agrega `deprecated: true` y una nota con el reemplazo. No se
  borra la entrada: hay bases viejas que la referencian.
- **Actualizar specs**: `ram_mb` se mide, no se copia de la model card.
- Cada cambio de catálogo va con su nota en `docs/CHANGELOG.md` y sube `updated`.
