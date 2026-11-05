# Argos — reescritura del RAG en Go (v2)

**Objetivo:** reemplazar el RAG en Python (`.rag/`, 565 MB) por un servicio en Go que indexe el vault
en Qdrant, con varias bases independientes, búsqueda cruzada entre ellas y control explícito de
recursos.

**Medido en la máquina:** vault de 1.4 GB y 7,874 archivos (6,674 `.md`, 257 `.pdf`, 294 `.png`);
RAG actual en Python con `multilingual-e5-large` (1024 dims), chunk 512/50, RRF k=60, MCP por stdio.

**Stack:** Go 1.27, Qdrant v1.20 (`qdrant/go-client`), ONNX Runtime directo (`yalue/onnxruntime_go`) + tokenizer (`sugarme/tokenizer`), MCP oficial, bbolt.

---

## Estado de implementación

- **Hecho:** módulo Go, `internal/config`, `internal/registry` (catálogo embebido y validado),
  `internal/models` (descarga con sha256, instalación atómica, verificación, borrado),
  `internal/embed` (ORT + pooling propio), `internal/version`, y `cmd/argosctl` con `init`, `paths`,
  `models list|catalog|show|install|use|remove|verify|sync` y `doctor`. Tests verdes, `vet` limpio.
- **Parsers:** Markdown, TXT, HTML, DOCX, EPUB y PDF (pdftotext), con tests por formato.
- **Manifiesto e ingesta:** bbolt + diff por chunk, reanudable, con progreso; probado de punta a punta
  contra Qdrant real (5 archivos → 28 puntos; segunda corrida 0 embeddings en 3.4 s). El manifiesto
  vive en `data/<colección>/manifest.db`, uno por base.
- **Multi-DB:** `db create|list|info|rm`, colección `argos_<nombre>` y manifiesto propio por base;
  `index --db N` resuelve origen, modelo, dims y colección.
- **Stack Docker completo:** `qdrant` + `argosd` + `argos-mcp` con la misma imagen (`argos:local`),
  modelos en el volumen `argos-home` (no horneados), healthchecks, `/metrics` y `stop_grace_period`.
  Medido: 48–56 ms en caliente; `stop` libera 2.4 GB.
- **MCP consolidado en el mismo módulo:** `cmd/argos-mcp` + `internal/mcp/{runner,jobs,daemon,tools}`;
  un solo repo y un solo `go test ./...`.
- **CLI con fast path:** `search` usa `argosd` cuando responde (66 ms total, 50 ms de motor).
- **Servicio `argosd`:** modelos residentes + `/v1/search` y `/v1/status`; 3.3 s la primera consulta
  (carga) y **58 ms** las siguientes. Concurrencia serializada por mutex en la sesión ONNX.
- **Híbrido:** dense + sparse (TF, `Modifier: Idf` en Qdrant) con `Prefetch` y fusión RRF. Los
  scores son RRF (≈0.016), no coseno: `--min-score` queda sin calibrar.
- **Cross-DB:** `--db all` en engine y CLI, fusión por rango entre bases.
- **Eval:** `argosctl eval` mide recall@k y MRR contra `testdata/eval.yaml` (30 consultas generadas
  desde títulos; el set hay que refinarlo a mano).
- **Búsqueda:** `search` con pool, tope por archivo, `--min-score`, salida JSON y reranker opcional.
- **Reranker:** cross-encoder bge-v2-m3 con plantilla `<s> q </s> </s> p </s>`, sigmoid y umbral.
- **Reranker medido:** 460 ms/par sin lote; 3.9 s con lote 8 y par truncado a 384 tokens.
- **Presets de hardware:** detección de GPUs por `/sys/class/drm` y selección de EP (CUDA/ROCm/OpenVINO)
  con `auto|cpu|gpu` por modelo; si el build de ORT no trae el EP, cae a CPU y lo reporta.
- **Docling:** cliente HTTP (`/v1/convert/file`) para PDFs sin capa de texto o con layout complejo,
  con caída a `pdftotext` si el sidecar no responde.
- **GPU: descartada en esta máquina.** ORT oficial no trae EP de GPU; OpenVINO ve el iGPU pero la
  compilación del modelo cuelga (Qwen3 >10 min, e5 >7 min). Ver [`SPIKES.md`](SPIKES.md).
- **Batching: descartado para CPU** (medido: 1 secuencia 843 ms, lote 4 1095 ms, lote 8 1141 ms).
- **Auditoría (ronda 2):** `index --path`, rechazo de bases anidadas, `doctor` verifica Qdrant y
  dims, claves de config para qdrant/index/layout, enlaces en DOCX, aviso al truncar modelos sin MRL,
  chequeo de espacio antes de descargar y manifiesto para documentos vacíos.
- **Auditoría de fases previas:** headings dentro de bloques de código ya no se parsean como
  secciones; la KV cache vacía se crea una vez por sesión; los puntos viejos se borran **después**
  del upsert; el payload conserva los tipos (enteros, no strings).
- **Spike A: PASA** — coseno 1.000000 contra la referencia Python; RSS 931 MB, 44 ms por texto.
  Ver [`SPIKES.md`](SPIKES.md).
- **Pendiente fase 0:** spike B (PDF 3 vías) y spike C (Docling).

---

## 1. Decisiones de arquitectura

| Decisión | Elección | Por qué |
|---|---|---|
| Vector DB | **Qdrant, fijo** | Es parte del producto, no un componente intercambiable: sin capa de abstracción de store |
| Embeddings | **e5-base** en CPU · Qwen3-0.6B como calidad | Medido: e5-base 128 ms/chunk vs 1121 ms de Qwen3. El decoder es 9× más lento sin GPU |
| Catálogo | `internal/registry/catalog.yaml` | Especificaciones, tiers, hashes y link oficial por modelo — ver [`MODELS.md`](MODELS.md) |
| Alternativas | `EmbeddingGemma-300M` (768, 2× más rápido), `bge-m3`, `e5-base` | Ya cargadas en el catálogo, no requieren cambios de código |
| Modelo como módulo | **Registry** con catálogo declarativo | Cambiar de modelo no toca código: una entrada en el catálogo y `models use` |
| Manifiesto | bbolt por base | Control de cambios, transaccional, sin CGO |
| Chunking | Estructura primero, **480 tokens** máx / **320 mín** (fusiona secciones cortas), 15 % solape · medido: 57,348 chunks de 350 tokens | El contexto explícito en el chunk es la mejora más barata (Anthropic: 35 % menos fallos solo con eso); recursive solo como fallback |
| PDF | `go-pdfium` (WASM, sin CGO) con fallback `pdftotext` | Motor de Chrome; se decide con una prueba sobre PDFs reales |
| IDs | UUIDv5(`rel#hash#ocurrencia`) | Idempotencia y sin colisión entre secciones idénticas |
| Reranker | **bge-reranker-v2-m3 int8** (560 MB, MIT), store aparte | Cross-encoder sobre el pool: sube la inclusión en el top final, no solo el orden |
| Transporte MCP | Streamable HTTP | En Docker no hay stdio |
| Bases | Una colección de Qdrant + `dbs/<nombre>/` con meta y manifiesto | Varias bases sin varias instancias de Qdrant |
| Vector | **512 dims (MRL)** + scalar int8 | 99 % del nDCG a la mitad de dims; int8 da 4× más con ~1 % de pérdida |

### Truncado Matryoshka

`index.dims` permite truncar el vector de Qwen3 (1024 → 512 → 256) y renormalizar. La evidencia:
MRL a 512 dims conserva ~99 % del nDCG@10, a 256 ~96 %. Es la palanca más barata para bajar memoria
y latencia de búsqueda, y va junto con una colección nueva (cambia el espacio vectorial).

### Lo que NO se abstrae

- **Qdrant no va detrás de una interfaz de proveedor.** Cada método del store termina siendo "lo que
  Qdrant ya hace" (`Prefetch`, `FusionQuery`, filtros por payload) y la abstracción se vuelve una
  copia peor de su API. Se usa el cliente directo; para tests hay un doble en
  `internal/store/qdranttest`.
- **Las bases son inmutables en su identidad**: nombre, modelo y dimensión no cambian después de
  crearse. Cambiar de modelo no muta la base: se crea una **generación nueva** (`nombre@2`) y se
  elige cuál responde. Reindexar sobre la misma generación es válido porque no cambia el esquema.

---

## 2. Componentes

Mapa completo con responsabilidades y reglas de dependencia: [`COMPONENTS.md`](COMPONENTS.md).

```
cmd/argosctl/        CLI (db, index, search, models, gpu, doctor, serve)
cmd/argosd/          servidor (REST + MCP)
internal/config      YAML + env + validación
internal/registry    catálogo de modelos con sha256, dims, archivos, runtime
internal/db          metadatos de bases: crear, listar, resolver nombre, generaciones
internal/vault       walk, filtros, hashing, detección de tipo
internal/parse       Parser por formato: markdown, pdf, docx, pptx, xlsx, epub, html, texto/código
internal/chunk       chunker estructural + fallback recursive
internal/embed       Embedder: ORT directo (pooling propio) y HTTP (Ollama/OpenAI)
internal/sparse      TF → índices/valores (el IDF lo aplica Qdrant)
internal/rerank      cross-encoder: puntúa (query, chunk) y reordena el pool
internal/store       cliente Qdrant + doble de test
internal/manifest    bbolt: archivo → hash, puntos, generación
internal/ingest      diff, lotes, progreso, reanudación, límites
internal/resources   política cpu/gpu/ram, detección de hardware
internal/search      densa, híbrida, fan-out entre bases, dedupe
internal/api         REST + SSE
internal/mcp         tools sobre el mismo backend
internal/progress    reporter TTY / JSON lines
internal/telemetry   slog + métricas
```

Dependencias en una sola dirección: `cmd → internal/{api,mcp,ingest,search} → dominio`; el dominio
no importa `store` ni `embed` concretos (el wiring vive en `cmd`).

Contratos principales:

```go
type Parser interface {
    Extensions() []string
    Parse(ctx context.Context, path string) (Document, error)
}

type Document struct {
    Path     string
    Title    string
    Meta     map[string]string
    Sections []Section   // heading + cuerpo, en orden
}

type Chunker interface {
    Chunk(doc Document, maxTokens, overlap int) ([]Chunk, error)
}

type Embedder interface {
    Dim() int
    Name() string
    MaxTokens() int
    EmbedPassages(ctx context.Context, texts []string) ([][]float32, error)
    EmbedQuery(ctx context.Context, text string) ([]float32, error)
}
```

---

## 3. Modelos como módulo

El catálogo, el estado instalado y el flujo de `models use` están especificados en
[`MODELS.md`](MODELS.md). Acá queda solo el contrato de código.

`internal/registry/catalog.yaml`, embebido con `embed.FS` y sobreescribible por el usuario en
`~/.config/argos/models.yaml`:

```yaml
models:
  - id: qwen3-embed-0.6b
    family: qwen3
    dims: 1024
    max_tokens: 32768
    pooling: last_token
    query_instruction: "Given a web search query, retrieve relevant passages that answer the query"
    files:
      - url: https://huggingface.co/<org>/qwen3-embed-0.6b-sbert-onnx/resolve/main/model.onnx
        sha256: "<se fija al agregar>"
        dest: model.onnx
      - url: .../tokenizer.json
        dest: tokenizer.json
    runtime: {kind: onnx, quant: int8}
    ram_mb: 900
  - id: e5-base
    family: e5
    dims: 768
    prefix: {query: "query: ", passage: "passage: "}
```

Reglas:

1. **El catálogo es dato, no código.** `argosctl models add <hf-repo>` calcula los sha256, escribe la
   entrada en `~/.config/argos/models.yaml` y descarga a `~/.local/share/argos/models/<id>/`.
2. **El embedder aplica sus propios prefijos o instrucciones.** El llamador nunca arma el texto con
   prefijos: si se olvida, la calidad cae sin error visible.
3. **Cada base fija su modelo al crearse** (`db create --model ...`), y `db info` lo muestra.
4. **Cambiar de modelo = nueva generación**, nunca mutar vectores de otro espacio.

---

## 4. Recursos: contenedor y aplicación

Son dos niveles distintos y no se mezclan.

### 4.1 Límites del contenedor (cgroups) — los pone la CLI

```sh
argosctl resources set  --cpus 4 --memory 2g [--pids 512] [--device /dev/dri]
argosctl resources show                     # límites reales + consumo
```

- Se guardan en el `.env` del stack (`ARGOS_CPUS`, `ARGOS_MEMORY`) y se aplican con `docker update`
  (cpus y memoria se cambian en caliente) o `docker compose up -d` cuando el cambio exige recrear.
- `resources show` lee `docker inspect` para el límite y `docker stats --no-stream` para el uso. Si
  alguien cambió el contenedor a mano, manda la realidad, no la config.
- **La CLI corre en el host**, no dentro del contenedor: habla con Docker por su socket y con el
  servicio por HTTP. Por eso puede administrar límites que desde adentro no se pueden tocar.
- Sin límite declarado, el contenedor usa todo el host. Los defaults son "sin límite", como pediste.

### 4.2 Ajustes de la aplicación (env)

```yaml
resources:
  mode: auto            # auto | cpu | gpu | mixed (elección de execution provider)
  devices:
    embed: auto
    rerank: auto
  threads: 0            # 0 = min(cpus del contenedor, físicos) - 1
  batch: 16
  pause_at_pct: 90      # guardián de RSS
```

- **Threads**: se resuelven contra los CPUs **del contenedor** (`/sys/fs/cgroup/cpu.max`), no contra
  los del host. Pedir 11 hilos con `--cpus 4` solo genera throttling.
- **RAM**: `GOMEMLIMIT` cubre el heap de Go, no la memoria nativa de ONNX. El guardián lee `VmRSS`
  cada 500 ms y, al pasar `pause_at_pct`, frena la admisión de archivos. Si hay límite de cgroup, el
  porcentaje se calcula sobre ese límite.
- **GPU**: se elige el execution provider al crear la sesión (OpenVINO para Intel, CUDA para NVIDIA,
  ROCm para AMD). Requiere `/dev/dri` montado y una build de ORT con ese EP; `doctor` lo informa.
- **VRAM**: no hay techo duro desde ORT; se controla con el tamaño de lote.

| Modo | Embeddings | Comportamiento |
|---|---|---|
| `auto` | CPU, salvo GPU dedicada usable | Default; en esta máquina resuelve CPU |
| `cpu` | Fuerza CPU | Threads = min(cgroup, físicos) - 1 |
| `gpu` | Fuerza GPU; falla si no hay backend | Para el server con dGPU |
| `mixed` | Densa en GPU, parseo en CPU | Solo con GPU dedicada |

Detección (`internal/resources/detect.go`) enumera **todas** las GPUs, sin asumir una:

1. `/sys/class/drm/card*/device/{vendor,device}` y `uevent` (`DRIVER=i915|xe|nvidia|amdgpu`).
2. Modelo legible por `lspci -mm -nn`.
3. Backends presentes: `vulkaninfo` o `/usr/share/vulkan/icd.d/*.json`, `nvidia-smi` + `/dev/nvidia0`,
   `/opt/rocm`, `libopenvino.so`.

```
ID  NOMBRE                    VENDOR   DRIVER   BACKENDS        RECOMENDADO
0   Intel UHD Graphics 620    Intel    i915     vulkan          no (integrada)
1   NVIDIA GeForce RTX 3060   NVIDIA   nvidia   cuda, vulkan    sí
```

Heurística explícita: dGPU > iGPU > CPU. Si la única GPU es Intel integrada, `auto` se queda en CPU:
comparte memoria y en un modelo de 0.6B suele perder contra 12 núcleos.

### Mixto: qué es real y qué no

- **A nivel de nodo, el mixto ya existe y es automático.** ONNX Runtime reparte el grafo por execution
  provider: lo que la GPU soporta va a la GPU, lo que no cae a CPU. No es un modo que se configure; es
  cómo funciona el runtime. El riesgo es el fallback silencioso: si la mitad del grafo cae a CPU, el
  rendimiento se degrada sin aviso. `doctor` reporta cuántos nodos quedaron en cada EP.
- **A nivel de modelo, sí es una decisión útil.** `devices.embed` y `devices.rerank` permiten, por
  ejemplo, embeddings en CPU (proceso por lotes, throughput) y rerank en GPU (una consulta, latencia).
  Eso es el "mixto" que aporta.
- **GPU sola con integrada: posible, casi nunca mejor.** El iGPU comparte el ancho de banda de la RAM
  del sistema y compite con el compositor; para modelos chicos suele perder contra 12 núcleos con
  AVX2. Se deja como opción (`devices.*: gpu`) y `auto` la evita en esta máquina. En el server con
  dGPU cambia el cálculo.

---

## 5. Bases (multi-DB)

```
~/.local/share/argos/
├── dbs/
│   ├── knowledge/{meta.json,manifest.db}
│   └── proyectos/{meta.json,manifest.db}
└── models/
```

Cada base = colección Qdrant `argos_<name>_g<gen>` + directorio de metadatos.

```sh
argosctl db create ~/Documents/Knowledge                # nombre: knowledge
argosctl db create ~/Documents/Github --name proyectos
```

1. Nombre por defecto: `slug(basename(ruta))`.
2. Si existe, se agrega un sufijo aleatorio de 4 caracteres (`knowledge-a3f9`) y se reporta el nombre
   final; `--name` lo define el usuario.
3. Se rechaza una ruta anidada dentro de una base ya registrada (evita indexar dos veces lo mismo).
4. Exclusiones globales: `node_modules`, `.git`, `.obsidian`, `.rag`, `venv`, `.venv`.
5. El modelo se fija al crear.

**Búsqueda cruzada:**

```sh
argosctl search "chunking markdown" --db all
argosctl search "qdrant" --db knowledge,proyectos -k 10
```

Fan-out por colección, top `k*3` de cada una, fusión RRF (k=60) sobre rangos, dedupe por
`(db, path, chunk_index)`. Cada hit lleva su `db`.

Después de fusionar se rerankea y se devuelven 20.

**Números de referencia (Anthropic, Contextual Retrieval):** recuperan 150, rerankean esos 150 y
pasan **20 chunks** al modelo. Probaron 5, 10 y 20: 20 fue el mejor. Sus métricas son recall@20.

Adopto esa forma: pool **100** → rerank → **20** al agente. Con chunks de 512 tokens son ~10k tokens
de contexto, aceptable para Hermes. Perfil rápido: `--rerank-depth 40 -k 8` (lo que tarda ~200 ms).

`--rerank auto|on|off` y `--rerank-depth N` (10-150) ajustan costo y latencia.

**Selección de tablas**: `--db` acepta una, una lista o `all`; `--model` restringe a las bases que
usan ese modelo; `--exclude-db` resta. La fusión por rango funciona entre modelos distintos porque no
compara scores, solo posiciones.

---

## 6. Formatos y parsing

Cómo lo resuelve la industria (2026): parseo por layout (Docling, Unstructured, Marker) cuando el PDF
importa; chunking **estructural** por defecto y recursive solo como fallback.

| Formato | Parser | Notas |
|---|---|---|
| `.md` | nativo | frontmatter → payload; headings → secciones |
| `.pdf` | `pdftotext` (poppler) primario; Docling para PDF complejo | < 200 chars/página → `needs_ocr` |
| `.docx` | unzip + `word/document.xml` | `w:pStyle` Heading1..3 → secciones |
| `.pptx` | unzip + `ppt/slides/slideN.xml` | `a:t`; título del slide como heading |
| `.xlsx` | `excelize` | hoja = sección; filas → tabla markdown |
| `.epub` | zip + XHTML + `goquery` | `nav` como estructura |
| `.html` | `goquery` | quitar nav/footer/script antes de extraer |
| `.txt`, código | nativo | lenguaje por extensión en el payload |

Decisión de PDF: **`pdftotext` primario**. Motivos:

- `pdfium` expone el texto crudo por posición y **no ordena la lectura**: en documentos a dos columnas
  devuelve los caracteres en orden aparentemente aleatorio. Su API deja el análisis al llamador.
- `pdftotext -layout` conserva la disposición visual pero infla el texto con espacios: un caso medido
  en la web da **96,8 % de whitespace** y un archivo 26× más grande, lo que arruina el conteo de
  tokens del chunk.
- `pdftotext` sin flag usa heurísticas de orden de lectura y es el que mejor se comporta en corpus
  mixtos; `-layout` se reserva para páginas con tablas.

El spike de la fase 2 ya se corrió: **gana `pdftotext` sin flag** (9–15 % de whitespace contra 19–35 %
de `-layout`, y ~10 % menos tokens). Ver [`SPIKES.md`](SPIKES.md). `-layout` queda para tablas.

---

## 7. CLI

La referencia completa (flags, ejemplos, exit codes y matriz de cobertura) está en
[`CLI.md`](CLI.md). Acá queda el resumen.

Dos binarios: `argosctl` (operación) y `argosd` (servicio). Todo soporta `--json` para el plugin.

```
argosctl db create <ruta> [--name N] [--model M] [--exclude GLOB]...
argosctl db list|info <N>|rm <N> [--force]
argosctl db reindex <N> [--new-model M]      # crea generación nueva
argosctl index [--db N|all] [--full] [--path P] [--dry-run] [--progress auto|plain|json]
argosctl search "<q>" [--db N|all|a,b] [--model ID]... [--exclude-db N]...
                      [-k 20] [--rerank auto|on|off] [--rerank-model ID] [--rerank-depth 100] [--json]
argosctl models list|add <repo>|use <id>|rm <id>
argosctl gpu list
argosctl resources show
argosctl doctor
argosctl serve [--addr 127.0.0.1:7700]
```

- `--progress` con TTY: barra con archivos, chunks/s, % y ETA. Sin TTY: JSON lines
  (`{"stage":"embed","done":1200,"total":6674,"pct":18}`), que el plugin consume sin parsear texto.
- Exit codes: 0 ok, 1 uso, 2 entorno (Qdrant/modelo ausente), 3 interrumpido.
- `doctor`: Qdrant alcanzable, colección existe, modelo descargado con sha256 correcto, threads
  efectivos, RAM disponible, GPUs detectadas.

---

## 8. Métricas y logs

No se duplica lo que da Docker (`docker stats`, stdout). Lo propio de la aplicación:

- `GET /v1/status`: bases, puntos por base, modelo, última corrida, progreso actual.
- `GET /metrics` opcional (Prometheus): `argos_files_total`, `argos_chunks_total`,
  `argos_embed_seconds`, `argos_rss_bytes`, `argos_queue_depth`.
- Logs `slog` JSON con `run_id`, `db`, `file`, `stage`, `ms`. La rotación es del driver de Docker.
- SSE en `/v1/jobs/{id}/events` para el plugin.

Regla: la app reporta **trabajo**, no salud del contenedor. `healthz` solo dice si Qdrant responde.

---

## 9. Pipeline de ingesta

```
walk → filtro → hash → diff → parse → chunk → embed (lote) → sparse → upsert → commit
```

- Diff **por chunk**, no solo por archivo: el manifiesto guarda el hash de cada chunk, así que al
  editar una nota solo se embeben los chunks nuevos y el resto conserva su vector. Es lo que hacen
  los pipelines incrementales en producción y evita re-embeber un archivo de 3,000 tokens por una
  línea cambiada.
- Diff por archivo: nuevo, modificado (borra los ids que desaparecieron), borrado del corpus.
- Índices de payload (`path`, `file_hash`, `mtime`) creados **antes** de subir datos, como recomienda
  Qdrant; sin ellos cada filtro y cada borrado hace scan.
- Escrituras en lote con `wait=false` y una espera final: `wait=true` por lote es 3–5× más lento.
- Commit por archivo, no por corrida: una interrupción no pierde el progreso.
- **Reanudación por manifiesto**: se confirma un archivo a la vez; una corrida interrumpida retoma desde el último archivo registrado (no hay journal aparte).
- Backpressure: canal acotado entre parse y embed.
- `index --path P` indexa solo ese archivo o directorio (mismo diff, filtrado por prefijo).
- `--dry-run` informa qué haría sin tocar Qdrant.

---

## 10. Fases

### Fase 0 — Esqueleto y spikes (1-2 días)
1. `go mod init`, `cmd/argosctl`, `cmd/argosd`, `Makefile`.
2. `internal/config` + tests.
3. **Spike A (camino A)**: `internal/embed/ort` con `onnxruntime_go` + `sugarme/tokenizer`, pooling
   de último token y normalización L2. Se valida contra un vector de referencia generado en Python
   con **coseno ≥ 0.999**; si no coincide, el bug es nuestro, no del modelo. Después se miden tiempo
   por chunk y RSS. `bge-m3` queda como red si el export no carga.
4. **Spike B**: 10 PDFs del vault por los tres extractores (`pdftotext`, `pdftotext -layout`,
   `go-pdfium`); se elige con caracteres útiles, whitespace y orden de lectura.
4b. **Spike C (layout)**: levantar `docling-serve` en CPU con un PDF a dos columnas y uno con tablas;
   medir tiempo, RAM y calidad del Markdown. Decide si el sidecar justifica sus 2 GB.
5. `internal/resources/detect` con tests sobre datos de `/sys` simulados.

### Fase 1 — Vault, parse y chunk (3 días)
6. `internal/vault`: walk con exclusiones + hash.
7. `internal/parse/markdown`: frontmatter + secciones.
8. Un parser por formato (docx, pptx, xlsx, epub, html, texto), con test cada uno.
9. `internal/chunk`: estructural con heading, 512 tokens, 15 % solape, tokenizer del modelo activo,
   ids UUIDv5.

### Fase 2 — Embeddings (2 días)
10. `internal/registry`: catálogo embebido + override; `models list/show` con el orden por tier
    (ver [`MODELS.md`](MODELS.md)).
10b. `models install`: descarga a `.tmp`, sha256, renombre atómico, progreso, chequeo de espacio.
10c. `models use`: instala si falta y escribe el default; con `--task reranker` resuelve el store
    `rerankers/`; `models remove` con las negativas por default/base que lo referencia.
10d. `models verify` y `models sync` (mantenedor: resuelve URLs y reescribe hashes).
11. `internal/embed/ort`: sesión ONNX con threads configurables, pooling de último token, L2, lotes,
    tests de similitud y comparación contra el vector de referencia.
12. `internal/embed/http`: Ollama/OpenAI-compatible con `httptest`.
13. Prefijos/instrucciones por familia; test de que el llamador no los agrega.

### Fase 3 — Store y bases (2-3 días)
14. `internal/store/qdrant`: colección dense+sparse, upsert, delete por `file_hash`, search.
15. `internal/db`: create/list/info/rm, slug, colisión con sufijo, rechazo de anidamiento.
16. `meta.json` + generaciones; test de que cambiar modelo crea `@2` sin tocar `@1`.

### Fase 4 — Ingesta (3 días)
17. `internal/manifest` (bbolt por base).
18. Diff + commit por archivo.
19. Lotes, reintentos, `--dry-run`, `--path`.
20. Guardián de RAM y backpressure; test con embedder lento simulado.
21. Progreso TTY/JSON; test de contadores.

### Fase 5 — Búsqueda y API (2 días)
22. Densa con filtros.
23. Fan-out multi-base + RRF con dos colecciones sembradas.
24. `/v1/search`, `/v1/status`, `/healthz` con `httptest`.
25. `/v1/jobs` + SSE.

### Fase 6 — MCP y CLI (2-3 días)
26. `kb_search`, `kb_get`, `kb_status`, `kb_reindex` con el cliente oficial.
27. `NewStreamableHTTPHandler`; config de Hermes a URL.
28. `argosctl models catalog` y `show`: tabla por tiers con specs y estado (contrato en
    [`CLI.md`](CLI.md)).
29. `argosctl db info|stats|docs`: ficha de la base, desglose y listado paginado.
30. `argosctl status`, `resources set`, `config get|set`, `version`, `completion fish`.
31. `--json` y exit codes consistentes en todos los comandos; test de la tabla de códigos.

### Fase 7 — Docker (1 día)
29. Dockerfile multi-stage (`golang:1.27` → `debian:13-slim`; ORT es glibc).
30. Compose con Qdrant, vault `:ro`, healthcheck, persistencia verificada.

### Fase 8 — Híbrido y calidad (2-3 días)
31. `internal/sparse` (TF) + `Modifier: Idf`.
31b. `internal/rerank`: cargar el modelo, puntuar 100 pares, reordenar; `--rerank off` para comparar.
31c. Matriz de medición con el eval set de 30 consultas — **3 perfiles × rerank on/off**:

    | Perfil | Pool | Devuelve | Qué se mide |
    |---|---|---|---|
    | rápido | 40 | 8 | recall@8, p50/p95, CPU |
    | default | 100 | 20 | recall@20, nDCG@10, p50/p95, CPU |
    | exhaustivo | 150 | 20 | recall@20, p50/p95, CPU |

    Regla: el perfil gana si mejora ≥3 puntos de recall sobre el mismo perfil sin rerank. Si el
    rerank no mueve la aguja en ninguno, se apaga (`--rerank off` como default) y queda como opción.
32. `Prefetch` + RRF; densa vs híbrida.
33. Set de 30 consultas con respuesta esperada; recall@5.
34. Benchmarks de indexación, incremental no-op y p95 de búsqueda.

---

## 9bis. Filtros de indexado

El default es **Markdown**; lo demás se habilita explícitamente.

```sh
argosctl index --dry-run                       # qué entraría con los filtros actuales
argosctl index --dry-run --include .md,.pdf    # solo esta corrida
argosctl config set filters.include .md,.pdf   # persistente
argosctl config set filters.exclude .png,.jpg
argosctl config set filters.exclude_dirs .git,.obsidian,node_modules
```

- Los filtros viven en `internal/vault`, no en el indexador: el mismo walk alimenta el dry-run, el
  conteo y el diff incremental.
- `--include` vacío significa "todo"; `--exclude` siempre gana.
- `--dry-run` no toca Qdrant: muestra archivos y MB por extensión.

## 10bis. Rutas y portabilidad (desktop ↔ server)

El estado **no vive dentro del corpus**: va en un directorio reservado, configurable.

```
${ARGOS_HOME:-~/.local/share/argos}/
├── data/       manifiesto y config runtime
├── models/     embeddings/ y rerankers/
└── qdrant/     storage de Qdrant
```

- `ARGOS_HOME` es la única raíz. Se puede mover a otro disco (`/mnt/datos/argos`) sin tocar el
  compose ni el código.
- El **vault** es otra cosa: `${VAULT_PATH}` montado `:ro`. Fuente, no estado.
- La CLI los administra:

```sh
argosctl init                                   # crea la raíz y el .env del stack
argosctl paths                                  # muestra raíz, textos, modelos, vault
argosctl config set paths.home  /mnt/datos/argos
argosctl config set paths.vault /mnt/vault
```

- `init` escribe el `.env` que consume el compose (`ARGOS_HOME`, `VAULT_PATH`) y crea los
  directorios con permisos correctos.
- Al mover el stack: copiar `$ARGOS_HOME` completo y ajustar `paths.vault`. Las rutas guardadas en
  `meta.json` son relativas al vault, así que el índice sigue válido.
- `doctor` verifica que la raíz exista, sea escribible y tenga espacio; y que el vault esté montado
  en modo lectura.

## 10ter. PDF complejo (layout-aware)

Los extractores de texto fallan en documentos con columnas, tablas o escaneos. Para eso el plan suma
un **parser por layout opcional**, como sidecar:

- `docling-serve` (imagen CPU oficial, ~4 GB de imagen, ~2 GB de RAM por worker, modelos incluidos).
- Argos lo llama por HTTP (`/v1/convert/`) y recibe Markdown con estructura y tablas.
- Se activa con `--layout auto|always|never`:
  - `auto`: rutea a Docling solo si el PDF parece complejo.
  - `always`: todo PDF por Docling.
  - `never`: solo `pdftotext`.
- **Heurísticas de ruteo** (sin modelo): proporción de líneas con gutter de 4+ espacios (columnas),
  densidad de texto por página, y si hay imágenes que cubren >60 % de la página (escaneo).
- Si Docling no responde, cae a `pdftotext` y lo deja registrado en el job; nunca falla la corrida.
- El sidecar vive en un `profile: layout` del compose: no arranca si no se pide.

Costo: Docling es Python + modelos, así que no entra en el binario Go. Es un servicio más, con su
propia métrica, y se puede dejar apagado en el desktop (donde sobran PDFs simples) y prendido en el
server.

## 11. Docker (destino)

`~/Documents/Docker/argos/compose.yaml`. El Dockerfile debe respetar: base Debian (no Alpine),
modelos en volumen, vault `:ro`, `data/` escribible, bind loopback.

---

## 12. Riesgos

| Riesgo | Mitigación |
|---|---|
| El export de Qwen3 no carga o el pooling no coincide | Resuelto: spike A en verde (coseno 1.000000) |
| Versión de ORT distinta entre indexado y consulta | Vectores de espacios distintos: pin 1.31 en host y Docker; `doctor` reporta la versión |
| Un modelo decoder en CPU es 9× más lento que un encoder | e5-base es el default; Qwen3 queda para GPU o indexado nocturno |
| RAM justa (3.5 GB libres, zram en uso) | Modelo int8, lote chico, guardián de RAM, cerrar Steam al indexar |
| iGPU Intel no acelera | `auto` resuelve CPU; GPU queda para el server |
| 257 PDFs con extracción pobre | Doble extractor + `needs_ocr` |
| Bases anidadas duplican puntos | Validación de prefijos al crear |
| Cambio de modelo silencioso | Colección por modelo+generación, visible en `db info` |

---

## 13. Criterios de aceptación

- `db create` + `index` de 6,674 markdown en menos de 60 min; segunda corrida < 5 s y 0 escrituras.
- `search --db all` fusiona rangos y devuelve el campo `db`.
- Archivo borrado → 0 puntos suyos.
- Cambio de modelo → generación nueva, base vieja intacta y consultable.
- `gpu list` enumera todas las GPUs con driver y backends.
- `--progress json` emite líneas parseables sin TTY.
- `go test ./...` verde, `gofmt -l .` vacío, `go vet` limpio.
