# argos

RAG del vault personal en Go: indexa Markdown y PDF en Qdrant y expone búsqueda por REST y MCP.

Reemplaza el RAG en Python que vive en `~/Documents/Knowledge/.rag/`.

- Plan de implementación: [`docs/PLAN.md`](docs/PLAN.md)
- Mapa de componentes: [`docs/COMPONENTS.md`](docs/COMPONENTS.md)
- Catálogo de modelos: [`docs/MODELS.md`](docs/MODELS.md)
- Roadmap (plugin y mejoras): [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Despliegue: `~/Documents/Github/argos/compose.yaml`
- Fuente de verdad: el vault. El índice es derivado y reconstruible.

## MCP

`argos-mcp` expone el CLI y el servicio como tools MCP (Streamable HTTP). Referencia completa en
[`docs/MCP.md`](docs/MCP.md).

```sh
./argos-mcp --argosctl ./argosctl --argosd http://127.0.0.1:7712
```

`search` va por `argosd` (modelos residentes); la administración sigue por el CLI. Unidades de
ejemplo en [`docs/systemd/`](docs/systemd/).

## Servicio

`argosd` mantiene los modelos cargados y sirve búsquedas por HTTP. El CLI carga el modelo en cada
comando (~3 s); el servicio responde en ~60 ms después de la primera consulta.

```sh
./argosd --addr 127.0.0.1:7712
curl -s -X POST http://127.0.0.1:7712/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"chunking","db":"knowledge","k":5}'
```

Unidad de ejemplo: [`docs/systemd/argosd.service`](docs/systemd/argosd.service).

## Uso rápido

```sh
make build
./argosctl init                 # crea ~/Services/argos y el .env del stack
./argosctl models catalog       # qué modelos hay, con specs y estado
./argosctl models sync e5-base   # fija los sha256 (descarga)
# si ya está instalado con --allow-unpinned:
./argosctl models sync --from-store e5-base
./argosctl models install e5-base bge-reranker-v2-m3-int8
./argosctl doctor
```

`models install` exige sha256 pinneado: `sync` es el paso que los calcula y los guarda en
`~/Services/argos/models.yaml`.

Estado: **fase 0 en curso** — CLI, config, catálogo y store de modelos funcionando; faltan los spikes
de ONNX, PDF y layout.
