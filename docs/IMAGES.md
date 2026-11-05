# Imágenes de Argos por hardware

Tres imágenes, un mismo código. Todas corren **solo CPU**; la diferencia es qué execution provider
llevan compilado y qué runtime de vendor incluyen.

| Tag | Base | ORT | GPU | Aceleración de CPU | Tamaño |
|---|---|---|---|---|---|
| `argos:nvidia` | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` | build oficial `gpu_cuda12` | NVIDIA (CUDA) | MLAS (AVX2/AVX512) | 8.7 GB (medido) |
| `argos:intel` | `debian:13-slim` + OpenVINO runtime | compilado `--use_openvino` | Intel iGPU/Arc (OpenVINO) | **OpenVINO CPU** (oneDNN) | ~2.5 GB |
| `argos:amd` | `rocm/dev-ubuntu-24.04` | compilado `--use_migraphx` | AMD (MIGraphX) | MLAS; ZenDNN opcional | ~8–10 GB |

Reglas:

1. **Las tres arrancan sin GPU.** ONNX Runtime siempre trae el EP de CPU; los providers cargan solo
   si se piden. Verificado con la imagen NVIDIA en un host sin NVIDIA: search funcionando.
2. **El EP se elige en caliente** por `resources.devices.embed` / `.rerank` (`auto|cpu|gpu`) y el
   `doctor` prueba cada EP de verdad (append a una sesión), no adivina por el vendor.
3. **Intel cubre CPU + iGPU en una imagen.** El EP OpenVINO se compila con un default
   (`--use_openvino CPU_FP32`) y el `device_type` se **overridea en runtime** con `GPU`/`AUTO`.
   Fuente: docs de ORT OpenVINO EP ("device_type ... overrides the accelerator hardware type ...
   at runtime").
4. **AMD ya no tiene EP ROCm.** Fue eliminado del árbol en ORT 1.31; AMD recomienda MIGraphX.
   Se compila desde fuente (`--use_migraphx`, ~45 min). El CPU de AMD usa MLAS por defecto; ZenDNN
   es opcional y agrega otra dependencia.

## Contrato de runtime

Variables que el contenedor espera:

| Variable | Valor | Quién la setea |
|---|---|---|
| `ARGOS_ORT_LIB` | `/usr/local/lib/libonnxruntime.so` | Dockerfile |
| `ARGOS_HOME` | `/home/argos/.local/share/argos` | Dockerfile |
| `ARGOS_QDRANT_URL` | `qdrant:6334` | compose |
| `ARGOS_VAULT` | `/vault` | compose |
| `ARGOS_IMAGE` | `argos:intel` etc. | CLI del sistema (`image: ${ARGOS_IMAGE:-argos:local}`) |

Dispositivos:

- **NVIDIA**: `docker compose -f compose.yaml -f compose.gpu.yml up -d` (nvidia-container-toolkit).
- **Intel/AMD**: montar `/dev/dri` (`devices: ["/dev/dri:/dev/dri"]`) y el usuario en el grupo
  `render`/`video` del host. Sin eso, el EP cae a CPU.

## Recetas de build

### NVIDIA (existente)

```sh
docker build --target runtime-gpu --build-arg ORT_FLAVOR=gpu -t argos:nvidia .
```

Usa el asset `onnxruntime-linux-x64-gpu_cuda12-1.31.0.tgz` y copia `libonnxruntime_providers_*.so`.

### Intel (pendiente de target)

1. Instalar OpenVINO runtime (deb oficial o tarball) en la imagen final.
2. Compilar ORT desde fuente:

```sh
./build.sh --config Release --build_shared_lib --parallel \
  --use_openvino CPU_FP32 --build_dir /build
```

3. Copiar `libonnxruntime.so*` y `libonnxruntime_providers_openvino.so`.
4. En runtime, `device_type=GPU` (o `AUTO:GPU,CPU`) para iGPU; `CPU` para CPU.

### AMD (pendiente de target)

1. Base con ROCm + MIGraphX.
2. Compilar ORT:

```sh
./build.sh --config Release --build_shared_lib --parallel \
  --use_migraphx --migraphx_home /opt/rocm --build_dir /build
```

3. Copiar `libonnxruntime.so*` y `libonnxruntime_providers_migraphx.so`.
4. Opcional: `--use_zendnn` para acelerar el CPU AMD (requiere ZenDNN/AOCL).

## Verificación por imagen

| Caso | Esperado |
|---|---|
| Host sin GPU | `/v1/status` provider `CPU`, search OK |
| Intel iGPU | provider `OpenVINO`, `doctor` lista el EP |
| AMD dGPU | provider `MIGraphX` |
| NVIDIA dGPU | provider `CUDA`, `nvidia-smi` visible dentro |
| CPU Intel | provider `OpenVINO` con `device_type=CPU` |
| CPU AMD | provider `CPU` (o ZenDNN si se compiló) |

## Límites

- No existe una imagen única para las tres marcas: CUDA, OpenVINO y MIGraphX son builds distintos de
  ORT (ORT permite combinarlos en un build, pero con todos los runtimes de vendor adentro: ~16 GB).
- Si el CPU es de una marca y la dGPU de otra (Intel + NVIDIA), la imagen elegida acelera la GPU y
  el CPU queda en MLAS. No hay doble aceleración salvo un build multi-EP a medida.
- Las imágenes de Intel/AMD todavía no están construidas; este documento es el contrato para la
  sesión que las agregue.
