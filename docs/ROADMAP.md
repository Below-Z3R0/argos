# Argos — roadmap

Lo que no entra en el plan actual. Nada de esto se implementa hasta que el plan esté cerrado y
corriendo.

## Plugin de UI (fase posterior)

Ideas para cuando exista el plugin, con lo que el backend tiene que exponer para soportarlas.

| Función | Necesita del backend |
|---|---|
| Árbol de bases con tamaño y cantidad de puntos | `GET /v1/dbs` con stats por base |
| Crear base apuntando a una carpeta | `POST /v1/dbs {path, name?, model?}` |
| Ver la DB: lista de documentos y chunks | `GET /v1/dbs/{name}/docs?offset&limit` |
| Inspeccionar un chunk | `GET /v1/dbs/{name}/chunk/{id}` |
| Buscar desde la UI | `POST /v1/search` (ya previsto) |
| Reindexar con progreso | `POST /v1/dbs/{name}/index` + SSE |
| Cambiar límites (CPU/GPU/RAM) | `PATCH /v1/resources` + `GET /v1/resources` |
| Ver GPU detectadas y elegir | `GET /v1/gpus` |
| Borrar o renombrar base | `DELETE`, `PATCH /v1/dbs/{name}` |

Regla: el plugin consume `--json` y SSE, nunca parsea logs.

## Mejoras de calidad

- **Reranker de segunda pasada**: reordenar 40 con texto truncado y volver a puntuar el top-8 con el
  texto completo (los rerankers de 8K lo permiten). Es la mejora que queda una vez que el reranker
  simple esté medido; primero hay que tener los números del actual.
- **OCR** para PDFs escaneados: hoy se marcan `needs_ocr` y se excluyen. Un pipeline aparte
  (tesseract u OCRmyPDF) los convertiría en texto indexable.
- **Chunking semántico** para documentos sin estructura (notas sueltas, correos).
- **Extracción de tablas** de PDF/XLSX como bloques estructurados.
- **Query rewriting**: expandir la consulta con sinónimos o con el LLM antes de buscar.
- **Watch mode** con `fsnotify` para reindexar al guardar.

## Operación

- Snapshot/restore de Qdrant si el índice deja de ser reconstruible rápido (hoy es derivado).
- Auth por token cuando el servicio deje de estar en loopback.
- Empaquetado del binario con `goreleaser`.

## No-goals

- Cambiar Qdrant por otro store: la DB es parte del producto.
- Embeddings por API como default: el camino crítico es local.
- Indexar imágenes sin OCR: no aportan texto.

## MCP

- Autenticación si el servidor deja loopback (token o Tailscale).
- `db_rm` y `models_install` como tools propias.
- `search_multi`: fan-out entre bases cuando el CLI lo soporte.
- Límite de concurrencia de jobs a nivel MCP (hoy lo corta el lock del manifiesto).
