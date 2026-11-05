# Despliegue en Docker

Decisión: **Qdrant, `argosd` y `argos-mcp` corren en contenedores**. Los modelos viven en un volumen,
no en la imagen.

```
┌────────────┐   gRPC   ┌──────────────┐   HTTP   ┌────────────┐
│  qdrant    │◀────────▶│   argosd     │◀────────▶│  argos-mcp │◀── agentes
│  (volumen) │          │ modelos en   │          │ CLI + MCP  │
└────────────┘          │ memoria      │          └────────────┘
                        └──────────────┘                 │
                              ▲                          │ exec
                              └──────── argosctl ◀───────┘
```

## Por qué los modelos van en un volumen

| Opción | Imagen | Arranque | Actualizar modelo |
|---|---|---|---|
| Hornear pesos | ~1.6 GB | inmediato, offline | rebuild de imagen |
| **Volumen (elegido)** | ~350 MB | descarga una vez | `models install` sin rebuild |

Los dos modelos por defecto pesan ~1.16 GB (Qwen3 585 MB + BGE reranker 545 MB). Google recomienda
hornear por debajo de 10 GB, pero en un proyecto que cambia código seguido, meter los pesos en la
imagen obliga a reconstruir y republicar 1.2 GB por cada cambio de una línea. El volumen evita eso y
sigue siendo offline después de la primera instalación.

## Ciclo de vida y memoria

- `docker compose stop argosd` → SIGTERM → `argosd` corta el servidor, cierra las sesiones ONNX y
  termina. Al morir el proceso, **el kernel libera su RSS**: no queda nada del modelo en memoria.
- `stop_grace_period: 30s` da margen a que terminen las búsquedas en curso.
- `docker compose start argosd` vuelve a cargar el modelo en la primera consulta (~3 s).
- La página de caché de los archivos `.onnx` puede quedar en el page cache del host; es reclaimable y
  no cuenta como memoria del contenedor.

## Métricas y estado

| Endpoint | Qué da |
|---|---|
| `:7712/metrics` | `argos_search_seconds`, `argos_search_total{db,model}`, `argos_hits_total`, `argos_rerank_seconds`, `argos_model_load_seconds{model}`, `argos_model_load_total{model,state}`, `argos_models_loaded{kind}` |
| `:7712/v1/status` | colección, modelo, `ready`, y **estado por modelo** (`not_loaded`, `loading`, `ready`, `error` + `error` + `load_ms`) |
| `:7712/healthz` | liveness: el proceso responde |
| `:7712/readyz` | readiness: Qdrant contesta |
| `:7711/metrics` | `argos_mcp_calls_total{outcome}`, `argos_mcp_search_total{source}` |
| `:7711/healthz` | liveness del MCP |

Errores: `argosd` responde JSON con el mensaje del engine y código 400/503/504; el estado del modelo
queda en `error` con el motivo, visible en `/v1/status`. Las tools MCP devuelven `isError: true` y el
texto del CLI.

## Operación

```sh
# construir y levantar
docker compose up -d --build

# instalar modelos (una vez; quedan en el volumen argos-home)
docker compose run --rm --entrypoint argosctl argosd models install --yes \
  qwen3-embed-0.6b bge-reranker-v2-m3-int8

# crear e indexar una base apuntando al vault montado
docker compose run --rm --entrypoint argosctl argosd db create /vault --name knowledge
docker compose exec argos-mcp argosctl index --db knowledge

# estado y métricas
curl -s localhost:7712/v1/status | jq
curl -s localhost:7712/metrics | grep argos_
curl -s localhost:7711/metrics

# apagar (libera la memoria de los modelos)
docker compose stop argosd argos-mcp
```

## Medido en esta máquina

| Métrica | Valor |
|---|---|
| Search en frío (carga del embedder) | 3.28 s |
| Search caliente | **48–56 ms** |
| `argosd` con embedder cargado | 917 MB |
| `argosd` con embedder + reranker | 2.33 GB |
| Memoria liberada al hacer `stop` | **~2.4 GB** |
| `argos-mcp` en reposo | 1.5–5 MB |
| Qdrant en reposo | 18 MB |

## Carga y descarga de modelos

`argosd` arranca perezoso. Para evitar los 3 s de la primera consulta:

```sh
argosd --preload embed     # none | embed | rerank | all
```

En caliente, sin reiniciar el contenedor:

```sh
argosctl service status            # estado por modelo
argosctl service load all          # embed + rerank
argosctl service unload all        # libera memoria
```

`/v1/models/load` y `/v1/models/unload` son la API; las tools MCP `models_load` y `models_unload`
hacen lo mismo. Un `unload` espera a que terminen las búsquedas en curso (RWMutex) y cierra las
sesiones ONNX.

**Memoria:** después de cerrar las sesiones se llama `malloc_trim(0)`, que devuelve las arenas
libres al sistema. Medido: 23 MB en reposo → 1.74 GB con ambos modelos → **139 MB tras `unload all`**;
segundo ciclo: 1.99 GB → 92 MB. No hace falta reiniciar el contenedor para recuperar la memoria.

## Docling (PDF escaneado)

Probado con la imagen `docling-serve-cpu:v1.37.0` y un PDF **sin capa de texto** (0 caracteres con
`pdftotext`, 281 KB): Docling lo convirtió y el índice quedó con **3 chunks** y el código Java
extraído, en 27 s (incluye carga del embedder). Docling en CPU usa ~2 GB de RAM.

```sh
docker compose --profile layout up -d
argosctl index --db scanned --include .pdf --path /ruta/al.pdf
```

`layout.mode: auto` rutea solo los PDFs con `needs_ocr`; `always` manda todos; `never` usa solo
`pdftotext`.

## Snapshots

```sh
argosctl snapshot create|list|delete <nombre>|restore <nombre> [--db N]
```

Las tools MCP son `snapshot_create`, `snapshot_list`, `snapshot_delete` y `snapshot_restore`. Los
archivos viven en el volumen `./qdrant-snapshots` (montado en `/qdrant/snapshots`), así sobreviven al
contenedor. Restaurar usa la ruta local del nodo, que funciona incluso si la colección ya no existe;
probado borrando la colección y recuperando los 7 puntos.

## Descarga de modelos

`models install` escribe `<modelo>/.state.json` mientras baja, con bytes y total. `models list`
muestra `descargando 45%` o `descarga incompleta`, y el CLI imprime porcentaje y MB. Si no hay red,
el error lo dice (`sin conexión o URL inaccesible`) y el estado queda visible; reintentar es seguro.

## Una sola imagen: CPU + NVIDIA (por defecto)

La imagen que usa el compose compila con `ORT_FLAVOR=gpu` sobre `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`
y **corre en cualquier host**:

- En CPU usa el provider CPU: `libonnxruntime.so` no enlaza CUDA y el provider CUDA solo se abre
  cuando se pide. Verificado: Qwen3 con la lib GPU en esta máquina sin NVIDIA, 39 ms/texto.
- En NVIDIA, `resources.devices.*: auto` detecta la tarjeta y appendea `CUDAExecutionProvider`.
- La decisión es en caliente, por modelo (`embed` y `rerank` se eligen por separado).

```sh
cd ~/Documents/Docker/argos
docker compose build                       # imagen unificada (~8.7 GB con cudnn)
docker compose up -d                       # CPU: funciona tal cual
docker compose -f compose.yaml -f compose.gpu.yml up -d   # NVIDIA: monta el driver
```

La imagen pesa 8.7 GB porque incluye el runtime CUDA completo. En un host que nunca va a tener
NVIDIA, `runtime-cpu` (416 MB) es la opción razonable.

Para un host sin GPU donde el tamaño importe, existe `runtime-cpu` (416 MB):

```sh
docker build --target runtime-cpu -t argos:cpu .
```

## AMD e Intel integrada: por qué no están en la misma imagen

No es una decisión de tamaño, es que **no hay un build de ORT que las cubra**:

- Los releases de ORT 1.31 solo publican assets **CPU** y **CUDA**. No hay `linux-x64-rocm`.
- Las notas de release lo dicen: **el EP ROCm fue eliminado del árbol**; AMD recomienda MIGraphX.
- MIGraphX exige compilar ORT desde fuente (`--use_migraphx`) contra el runtime ROCm (~45 min) y
  produce **otro núcleo** de ORT. No se puede cargar en la misma `libonnxruntime.so` que CUDA.
- Lo mismo pasa con OpenVINO (Intel iGPU): requiere un build propio con `--use_openvino`.

Una imagen con CUDA + MIGraphX tendría dos núcleos de ORT y ~8 GB de runtimes de vendor. La vía
razonable si algún día hay AMD es un target aparte (`runtime-amd`) compilado desde fuente; mientras
tanto, esa máquina corre en CPU con la misma imagen.

`doctor` ahora verifica esto de verdad: appendea cada EP a una sesión y reporta cuáles existen en la
build, en vez de adivinar por el vendor de la tarjeta.

## Trade-offs y límites de este enfoque

1. **cgo en la imagen**: ORT 1.31 y `libtokenizers` se bajan y compilan en el builder. El build tarda
   unos minutos; el runtime es `debian:13-slim` (glibc, no Alpine).
2. **Descarga inicial**: el volumen nace vacío. En un servidor sin salida a internet hay que
   pre-sembrar `argos-home` (copiar `models/` desde otra máquina).
3. **GPU**: la imagen es CPU. Una dGPU necesita una build de ORT con CUDA/ROCm y montar `/dev/dri`;
   `resources.devices.*` ya elige el EP, pero el binario debe traerlo.
4. **`argosctl` dentro del contenedor**: las tools de administración del MCP ejecutan el CLI adentro,
   así que las rutas son las del contenedor (`/vault`). El CLI del host sigue sirviendo para lo que
   necesite Docker (`resources set`, futuro), porque el socket no está montado.
5. **Dos contenedores, un volumen**: `argosd` y `argos-mcp` comparten `argos-home`. El manifiesto
   bbolt bloquea la segunda indexación concurrente; por diseño hay un solo job de index a la vez.
6. **MCP sin auth**: publica en loopback. Si se expone, token estático o Tailscale.
