# Imágenes de Argos por hardware

Argos se construye en **dos capas**:

```text
b3l0w-ort:<variante>-1.31     runtime + ONNX Runtime + execution provider
        ↓
argos:<variante>-<sha>        binarios Go (docker/app.Dockerfile)
```

La capa de app no recompila ONNX Runtime. Actualizar Argos es un build de segundos/minutos.

## Bases publicadas

| Variante | Base | Provider | Notas |
|---|---|---|---|
| `cpu` | `ghcr.io/below-z3r0/b3l0w-ort:cpu-1.31` | CPU | Debian slim; imagen de app ~416 MB. |
| `cuda` | `ghcr.io/below-z3r0/b3l0w-ort:cuda-1.31` | CUDA | Runtime CUDA/cuDNN; necesita driver NVIDIA en el host. |
| `intel` | `ghcr.io/below-z3r0/b3l0w-ort:intel-1.31` | OpenVINO | OpenVINO 2026.4.1; `/dev/dri` para iGPU. |
| `amd` | `ghcr.io/below-z3r0/b3l0w-ort:amd-1.31` | MIGraphX | ROCm 7.2 / MIGraphX 2.15; `/dev/dri`. |

Todas arrancan en CPU aunque no haya GPU. El provider se selecciona por
`resources.devices.embed` / `.rerank` (`auto|cpu|gpu`) y `doctor` prueba el EP real.

## Construir la app

La base tiene que estar publicada o presente localmente:

```sh
# CPU (default)
make image-cpu

# Otra variante
make image-cuda
make image-intel
make image-amd

# Equivalente directo
docker build -f docker/app.Dockerfile \
  --build-arg BASE_IMAGE=ghcr.io/below-z3r0/b3l0w-ort:cpu-1.31 \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  -t argos:local .
```

`Makefile` ya no usa el Dockerfile unificado ni `runtime-*`/`ORT_FLAVOR`. El `Dockerfile` de la raíz
es legacy y no se usa en el camino actual.

## Runtime

| Variable | Valor | Quién la setea |
|---|---|---|
| `ARGOS_ORT_LIB` | `/usr/local/lib/libonnxruntime.so` | base `b3l0w-ort` |
| `ARGOS_HOME` | `/home/b3low-z3r0/Services/argos` | compose (bind mount, misma ruta dentro y fuera) |
| `ARGOS_QDRANT_URL` | `qdrant:6334` | compose |
| `ARGOS_VAULT` | `/home/b3low-z3r0/Documents/Knowledge` | compose (misma ruta que el host) |
| `ARGOS_SERVICE_URL` | `http://argosd:7712` | compose del MCP |

Dispositivos:

- **NVIDIA**: `docker compose -f compose.yaml -f compose.gpu.yml up -d` con nvidia-container-toolkit.
- **Intel/AMD**: montar `/dev/dri` (`devices: ["/dev/dri:/dev/dri"]`) y estar en el grupo
  `render`/`video`. Sin eso, el EP cae a CPU.

## Distribución

Las imágenes se publican en GHCR:

```text
ghcr.io/below-z3r0/b3l0w-ort:{cpu,cuda,intel,amd}-1.31
ghcr.io/below-z3r0/argos:{cpu,cuda,intel,amd}-<sha>
ghcr.io/below-z3r0/argos:{cpu,cuda,intel,amd}-latest
```

El CLI del sistema puede hacer `docker pull` y escribir `ARGOS_IMAGE` en el compose; no necesita
compilar. `latest` se publica solo después del smoke test.

## Verificación

| Caso | Esperado |
|---|---|
| Host sin GPU | provider `CPU`, search OK |
| NVIDIA | provider `CUDA` |
| Intel iGPU | provider `OpenVINO` |
| AMD | provider `MIGraphX` |
| Qdrant caído | `/readyz` falla, `/healthz` sigue OK |

## Pendiente

- `argos:amd-*`: la base `b3l0w-ort:amd-1.31` está publicada, falta la app.
- `argos:*-arm64`: no hay base arm64 publicada todavía.
- La app Intel está publicada desde `b4bfc88`; la corrida actual de `app.yml` solo reconstruyó CPU/CUDA.
