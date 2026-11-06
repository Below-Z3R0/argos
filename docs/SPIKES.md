# Spikes

Resultados de las pruebas que deciden el diseño. Cada spike se corre una vez, se mide y se anota
acá para no volver a discutirlo de memoria.

## A — Embeddings ONNX (Qwen3-Embedding-0.6B) · **PASA**

**Qué se probó:** cargar el export de `onnx-community/Qwen3-Embedding-0.6B-ONNX` (int8, 585 MB) con
`onnxruntime_go`, tokenizar con `daulet/tokenizers`, aplicar **pooling de último token** y
normalización L2, y comparar contra una implementación de referencia en Python.

**Resultado:** coseno **1.000000** en los tres textos, contra un umbral de 0.999.

| Métrica | Valor |
|---|---|
| Carga del modelo (fría) | 2.9 s |
| Embedding de textos cortos (~22 tokens) | 44 ms por texto |
| RSS pico del proceso | 931 MB |
| Presupuesto del catálogo | 900 MB |

El RSS medido coincide con el presupuesto: el número del catálogo no estaba inflado.

### Hallazgo importante: la versión de ONNX Runtime cambia los vectores

La primera comparación dio **0.960** con la misma referencia. La causa no era el pooling: era
**onnxruntime 1.19 en Python contra 1.31 en Go**. Al alinear la versión (1.31 en ambos), el coseno
pasó a 1.000000.

Con un modelo int8, los kernels de cuantización cambian los valores. Consecuencia operativa:

- **Indexar y consultar tienen que usar la misma build de ORT.** Si el índice se generó con 1.31 y
  las consultas corren con otra versión, los vectores no son del mismo espacio.
- La imagen Docker pinnea **ORT 1.31** y **libtokenizers 1.27**; el binario del host, también.
- `doctor` debe reportar la versión del runtime, no solo la del modelo.

### Hallazgo secundario: el tokenizer

`sugarme/tokenizer v0.3.0` no puede parsear el `tokenizer.json` de Qwen3 (`NewTokenizerFromFile` es
un stub con `TODO: implement`, y `pretrained.FromFile` entra en pánico). Se usa
`daulet/tokenizers` (binding a la librería Rust de HuggingFace), que es la misma que generó el
archivo. Requiere `libtokenizers.a` (14 MB, release v1.27.0) y cgo.

### Detalle del grafo

El export incluye la KV cache: 56 entradas `past_key_values.*` y 56 salidas `present.*`. Para
embeber se alimentan 56 tensores vacíos de forma `[1, 8, 0, 128]` y se ignora todo menos
`last_hidden_state`.

### Reproducir

```sh
export CGO_LDFLAGS="-L$HOME/Services/argos/runtime/libtokenizers"
ARGOS_MODEL_DIR=$HOME/Services/argos/models/embeddings/qwen3-embed-0.6b \
ARGOS_ORT_LIB=$HOME/Services/argos/runtime/onnxruntime-linux-x64-1.31.0/lib/libonnxruntime.so \
go test ./internal/embed -run TestGoldenVectors -v
```

El test compara contra `internal/embed/testdata/golden_vectors.json`, que son los vectores de la
referencia Python. Un cambio de tokenizer o de pooling lo rompe.

### Throughput: decoder vs encoder en CPU

Medido con textos de ~512 tokens (10 repeticiones), mismo runtime y mismas condiciones:

| Modelo | Arquitectura | Por chunk | RSS pico | 114,643 chunks |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B | decoder + KV cache | **1121 ms** | 1.58 GB | **~35.7 h** |
| multilingual-e5-base | encoder (BERT) | **128 ms** | 778 MB | **~4.1 h** |

El corpus real, medido con `index --dry-run --chunks`:

| Configuración | Chunks | Tokens/chunk | Tokens totales | Estimado Qwen3 | Estimado e5 |
|---|---|---|---|---|---|
| Un chunk por sección | 114,643 | 175 | 20.1 M | 8.0 h | 1.4 h |
| **Fusión 480/320** | **57,348** | **350** | 20.1 M | **8.0 h** | **1.4 h** |
| Fusión 800/500 (default) | 36,397 | 542 | 19.7 M | 7.9 h | 1.4 h\* |
| Fusión + PDFs (800/500) | ~49,000 | ~570 | 28.0 M | ~11.2 h | ~1.9 h |

\* e5-base tiene 512 tokens de límite: con chunks de 542 promedio **trunca**, así que la fila de 800
solo es válida con Qwen3.

### Optimización: no pedir los tensores de KV

El export devuelve 56 salidas `present.*` que nadie lee. Pidiendo **solo `last_hidden_state`**,
ONNX Runtime poda ese subgrafo:

| Métrica | Todas las salidas | Solo last_hidden_state |
|---|---|---|
| Tiempo por chunk de 512 tokens | 1121 ms | **716–738 ms** (-35 %) |
| RSS pico | 1.58 GB | **1.16 GB** (-27 %) |
| Vectores | — | idénticos (golden test verde) |

Nota de medición: con la máquina cargada (varios runs seguidos, load average > 5) el mismo chunk
tarda ~1050 ms. Las cifras de esta tabla son en reposo; el número a usar para estimar es 716 ms.

**Corrección importante:** fusionar reduce la cantidad de chunks pero **no** el tiempo de embedding,
porque el costo depende de los tokens y el corpus tiene los mismos tokens se corte como se corte.
Lo que se ahorra es cantidad de puntos (memoria de Qdrant, tamaño del índice) y sobrecarga por chunk.
La palanca real de tiempo es batching o GPU.

### Por qué fusionar secciones cortas

- **NVIDIA (2024, cinco datasets):** los chunks de 128 tokens fueron los peores (0.421 en KG-RAG);
  512–1024 rindieron mejor, y 2,048 empeoró por ruido. Hay un óptimo en el medio.
- **FloTorch/Vecta (2026, 50 papers):** recursive 512 fue la mejor estrategia end-to-end (69 %),
  fixed 512 (67 %), fixed 1024 (61 %). Semantic chunking quedó última (54 %).
- **Anthropic (Contextual Retrieval):** sus números asumen chunks de **800 tokens** y pasar 20 al
  modelo.
- **Patrón small-to-big / parent-document (LangChain, LlamaIndex, LanceDB):** recuperar con unidades
  chicas y devolver contexto más grande es mejor que fragmentar; un chunk de 175 tokens es un
  fragmento, no una unidad semántica.

Consenso: **400–512 tokens con 10–20 % de solape**. Nuestro 350 promedio con máximo 480 queda dentro
de ese rango; el siguiente ajuste, si hace falta, es subir MinTokens a 400.

Conclusión: en CPU, un decoder con KV cache cuesta ~9× más por chunk y ~2× más de RAM. Para 114k
chunks ninguna de las dos tasas es cómoda: Qwen3 son 35 horas y e5-base 4.

Dos palancas antes del primer indexado completo: **fusionar secciones cortas** (el promedio de 175
tokens es bajo; apuntar a ~400 bajaría a ~50k chunks) y **batching** en la fase 4.

Efecto en el chunking: e5-base tiene 512 tokens de límite **incluyendo prefijos y especiales**, así
que el chunk baja a 480.

## B — PDF (pdftotext vs `-layout`) · **PASA: gana plain**

10 PDFs reales del vault, ambos modos, mismo runtime:

| Métrica | `pdftotext` | `pdftotext -layout` |
|---|---|---|
| Whitespace | **3–15 %** | 19–35 % |
| Líneas con gutter de 4+ espacios | **0** | 0–150 |
| Tokens (libro de 113 k chars) | **32,260** | 33,755 (+4.6 %) |
| Tokens (reglamento chico) | **179** | 186 (+3.9 %) |
| Orden de lectura | correcto | correcto, pero con padding |

`-layout` no mejora el orden de lectura: preserva posiciones, y el precio es 10–20 % más de tokens y
hasta 35 % de whitespace. El chunker terminaría contando padding como contenido.

**Decisión: `pdftotext` sin flag es el extractor default.** `-layout` queda para páginas con tablas
(donde el padding ayuda a reconstruirlas) y Docling para PDFs complejos.

`go-pdfium` no entra: no hay módulo de backend Go publicado (`go-pdfium-wazero` no existe) y pdfium
no ordena la lectura por sí solo, que es justo el problema a resolver.

### Nota del corpus

257 PDFs suman **1,011 MB** (promedio 3.9 MB). Con el default `.md` quedan afuera; para incluirlos:
`argosctl config set filters.include .md,.pdf`.

## C — Docling (layout-aware) · pendiente

## D — Batching en CPU · **NEGATIVO**

Mismo texto (10 chunks de 512 tokens), mismo runtime, máquina en reposo:

| Batch | ms/chunk | RSS pico |
|---|---|---|
| **1** | **843** | **1.19 GB** |
| 4 | 1095 | 2.02 GB |
| 8 | 1141 | 2.91 GB |

El batching **empeora** en CPU: ORT ya usa los 12 hilos en una sola secuencia, y el padding de lotes
mixtos hace trabajo de más. Coincide con el reporte de Manticore ("throughput lower than one-by-one
because of the padding tax"). Default: `resources.batch: 1`. La palanca real es GPU o un modelo
encoder, no el lote.

## E — GPU en esta máquina (Intel iGPU) · **NEGATIVO**

| Intento | Resultado |
|---|---|
| ONNX Runtime 1.31 oficial | Solo trae `CPUExecutionProvider`; no hay OpenVINO EP |
| `onnxruntime-openvino` (pip) | No existe wheel para Python 3.14 |
| OpenVINO 2026.4.1 + `openvino-intel-gpu-plugin` | Detecta `['CPU', 'GPU']` |
| Qwen3 (59 entradas, KV cache) | `compile_model` colgado >10 min; proceso matado |
| e5-base (encoder puro) en OpenVINO | CPU: 1302 ms/chunk (peor que ORT: 128 ms). GPU: compilación colgada >7 min |

Conclusión: en esta máquina el iGPU (UHD Gen9) no es una vía usable para estos modelos. El batching no
tiene dónde rendir; el camino práctico sigue siendo CPU con ORT, o una dGPU en el server.

Los paquetes `openvino` y `openvino-intel-gpu-plugin` quedaron instalados (~500 MB) solo para este
test; se pueden quitar con `pacman -R openvino openvino-intel-gpu-plugin`.

## F — Reranker (bge-reranker-v2-m3 int8)

Sanidad semántica con la plantilla `<s> q </s> </s> p </s>`:

| Par | Score |
|---|---|
| "¿qué es un gato?" ↔ "Un gato es un animal doméstico…" | 0.978 |
| "¿qué es un gato?" ↔ "El índice bursátil subió…" | 0.000 |

Costo en CPU (el reranker es un encoder XLM-R de 568M, no un bi-encoder):

| Configuración | Tiempo |
|---|---|
| 15 pares de ~512 tokens, sin batch | 6.9 s (~460 ms/par) |
| Con batch 8 y par truncado a 384 tokens | 3.9 s (incluye 2.3 s de carga) |

Consecuencia: `--rerank-depth` no es gratis. El default rerankea todo el pool (`k × 3`); con `k=20`
son 60 pares y varios segundos. Para consultas interactivas conviene `--k 5` o `--rerank-depth 20`.
