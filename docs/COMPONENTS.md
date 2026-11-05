# Argos — componentes

Mapa de paquetes, responsabilidades y reglas de dependencia. El detalle de comportamiento está en
[`PLAN.md`](PLAN.md); esto es la estructura.

## Capas

```
┌──────────────────────────────────────────────────────────────┐
│ cmd/argosctl        cmd/argosd                               │
└───────────────┬──────────────────────────────┬───────────────┘
                │                              │
┌───────────────▼──────────────┐  ┌────────────▼───────────────┐
│ internal/api   internal/mcp  │  │ internal/jobs              │
└───────────────┬──────────────┘  └────────────┬───────────────┘
                │                              │
┌───────────────▼──────────────┬───────────────▼───────────────┐
│ internal/search              │ internal/ingest               │
└───────┬──────────┬───────────┴───────┬──────────┬────────────┘
        │          │                   │          │
┌───────▼───┐ ┌────▼─────┐ ┌───────────▼──┐ ┌─────▼─────┐
│ store/    │ │ embed    │ │ parse  chunk │ │ manifest  │
│ qdrant    │ │ sparse   │ │ vault        │ │ db        │
└───────┬───┘ └────┬─────┘ └──────────────┘ └─────┬─────┘
        │          │                              │
┌───────▼──────────▼──────────────────────────────▼───────────┐
│ internal/resources  internal/registry  internal/models      │
│ internal/config     internal/telemetry internal/progress    │
│ internal/types      internal/version                        │
└─────────────────────────────────────────────────────────────┘
```

Regla: una flecha apunta solo hacia abajo. `types` no importa nada del proyecto; ningún paquete de
dominio importa `cmd`; `api` y `mcp` no contienen lógica, solo traducción.

## Componentes

| Paquete | Responsabilidad | Expone | Depende de |
|---|---|---|---|
| `internal/types` | Structs puros compartidos, hoy `Point` | Tipos | stdlib |
| `internal/config` | Cargar YAML + env, defaults, validación, rutas XDG, escritura atómica | `Config`, `Load`, `Save` | types |
| `internal/version` | Build info (`commit`, `date`, `version`) | `Info()` | — |
| `internal/telemetry` | `slog` JSON, run IDs, contadores | `Logger`, `Metrics` | config |
| `internal/progress` | Reporter de progreso: TTY, JSON lines, SSE | `Reporter` | types |
| `internal/resources` | Detección de CPU/GPU/RAM; resolución de política; límites (threads, GOMEMLIMIT, guardián RSS); opciones de sesión ORT | `Detect`, `Resolve`, `Guard` | types, config |
| `internal/registry` | Catálogo embebido + override, validación de esquema, resolución de modelo, tiers | `Catalog`, `Get`, `Default` | types, config |
| `internal/models` | Ciclo de vida de artefactos: descarga, sha256, `.tmp`, renombrado atómico, lock, espacio en disco, estado instalado | `Install`, `Remove`, `Verify`, `Installed` | registry, config |
| `internal/db` | Metadatos de bases: slug, sufijo por colisión, crear/listar/borrar, `meta.json` y manifiesto por base | `Manager` | — |
| `internal/vault` | Recorrido con exclusiones, detección de tipo, hash, rutas relativas estables | `Walk`, `Hash` | types |
| `internal/parse` | `Parser` por formato (md, txt, html, docx, epub, pdf) + registro por extensión; ruteo de PDF complejo; `needs_ocr` | `Parser`, `Registry` | types |
| `internal/parse/layout` | Cliente HTTP de Docling (`/v1/convert/`): PDF complejo → Markdown | `Converter` | types, config |
| `internal/chunk` | Chunking estructural (headings), solape, conteo de tokens, IDs deterministas | `Chunker` | types, registry |
| `internal/embed` | `Embedder`; implementaciones ORT (pooling propio) y HTTP; lotes, reintentos, prefijos | `Embedder`, `NewORT`, `NewHTTP` | types, registry, resources |
| `internal/sparse` | TF → índices/valores para el vector sparse | `Encoder` | types |
| `internal/rerank` | Cross-encoder: puntúa (query, chunk) y reordena el pool fusionado | `Reranker` | types, registry, resources |
| `internal/store/qdrant` | Colecciones, upsert, delete por `file_hash`, scroll, count, búsqueda densa e híbrida | `Store` | types |
| `internal/store/qdranttest` | Doble en memoria para tests sin Qdrant | `Store` | types |
| `internal/manifest` | bbolt por base: archivo → hash, mtime, ids, generación; transacciones | `Manifest` | types, config |
| `internal/ingest` | Diff, plan, workers, backpressure, journal, reanudación, dry-run | `Run`, `Plan` | casi todo lo anterior |
| `internal/search` | Embebe la consulta, fan-out por base/modelo, RRF, dedupe, filtros | `Search` | types, embed, store, db |
| `internal/jobs` | Jobs de indexación en background: estado, progreso, cancelación | `Manager` | ingest, progress |
| `internal/api` | REST + SSE; validación de request; mapeo de errores | `NewServer` | search, jobs, db |
| `internal/mcp` | Tools MCP sobre los mismos servicios | `Register` | search, jobs, db |
| `cmd/argosctl` | Parseo de flags, salida tabla/JSON, exit codes, subcomandos | binario | todos |
| `cmd/argosd` | Wiring del servidor: config → registry → store → ingest → api/mcp | binario | todos |

## Flujo de indexación

```
vault.Walk ──► parse.Registry ──► chunk.Chunker ──► embed.Embedder ──► sparse.Encoder
                    │                    │                  │
                    ▼                    ▼                  ▼
              types.Document       types.Chunk        []float32
                    │                    │                  │
                    └──────────► ingest.Run ◄──────────────┘
                                     │
                     ┌───────────────┼────────────────┐
                     ▼               ▼                ▼
              manifest.Manifest  store.Store   progress.Reporter
```

`ingest` es el único que conoce el orden; los demás no saben que existe un pipeline.

## Flujo de búsqueda

```
search.Search(query, db)
   │
   ├─► db.Manager      (qué bases y con qué modelo)
   ├─► embed.Embedder  (una vez por modelo distinto)
   ├─► store.Store     (prefetch dense + sparse por colección)
   ├─► fusión RRF + dedupe por archivo
   └─► rerank.Rerank(pool 40) → top 8 → []types.Hit
```

## Reglas de implementación

1. **Sin estado global.** Nada de singletons: todo se construye en `cmd` y se pasa por constructor.
2. **`context.Context` en toda operación que pueda bloquear** (I/O, red, ORT, Qdrant).
3. **Errores con tipo**: `ErrNotFound`, `ErrInvalidPath`, `ErrModelMissing`, para que la CLI mapee a
   exit codes sin comparar strings.
4. **Interfaces solo donde hay dos implementaciones reales o un doble de test**: `Parser`,
   `Embedder`, `Store`. El resto son structs concretos; abstraer "por si acaso" es lo que produce
   capas inútiles.
5. **`internal/parse` crece por formato, no por caso**: agregar `.odt` es un archivo nuevo y una
   línea en el registro.
6. **Un paquete, un motivo para cambiar.** Si `ingest` empieza a saber de formatos o de Qdrant, la
   lógica está en el lugar equivocado.

## Lo que no es un componente

- Qdrant: proceso externo.
- Docker: empaquetado.
- El plugin: cliente de la API.
