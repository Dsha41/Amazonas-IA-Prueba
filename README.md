# SERVER-IA Amazonas

Servidor FastAPI + WebSocket de inferencia de visión por computadora para
**detección de personas con analítica retail** (demografía género/edad,
face re-identification, conteo único, atención vendedor-cliente,
monitoreo de stock).

Es el backend de inferencia del cliente
[Amazonas View / windows_managers_view](../windows_managers_view) — el cliente
envía frames por WebSocket y recibe la imagen anotada + metadata en JSON.

## Arquitectura

```
Cliente (Amazonas View / windows_managers_view)
      │
      │  WebSocket  ws://<host>:9000/ws/Personal%20de%20Amazonas
      │  (msgpack o JSON; frames base64/bytes)
      ▼
SERVER-IA  (este repo)
      │
      ├─ FastAPI + uvicorn         (src/app/app.py)
      ├─ ClientWorker por cliente  (cola asyncio, drop-head)
      └─ PersonAmazonas            (src/analityc/core/person_amazona_inference.py)
            │
            ├─ YOLO11m  → detección de personas (COCO clase 0)
            ├─ BoTSORT  → tracking estable con Kalman + IoU
            ├─ YuNet    → detector facial + 5 keypoints
            ├─ InsightFace genderage.onnx  → género/edad ensemble
            ├─ ArcFace  w600k_r50.onnx     → embeddings 512-d para Re-ID
            └─ Caffe age/gender_net        → fallback ensemble
```

## Requisitos

- Python **3.10+** (probado en 3.11 y 3.12)
- Windows / Linux. Las rutas del código son agnósticas al SO.
- (Opcional pero recomendado) GPU NVIDIA con CUDA 11.8 y drivers actualizados.
- **Git LFS** instalado para descargar los pesos del repo.

## Instalación

### 1. Clonar con LFS

```bash
git lfs install              # solo la primera vez
git clone <url-del-repo> SERVER-IA
cd SERVER-IA
git lfs pull                 # baja los .onnx, .caffemodel, .pb
```

### 2. Crear entorno virtual + instalar dependencias

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

> Sin GPU: edita `requirements.txt` cambiando `torch==2.7.1+cu118` por
> `torch==2.7.1`, `torchvision==0.22.1`, y `onnxruntime-gpu` por
> `onnxruntime`.

### 3. Descargar pesos restantes

El repo incluye (vía Git LFS):
- `models/classifiers/genderage.onnx`         (InsightFace, 1.3 MB)
- `models/classifiers/w600k_r50.onnx`         (ArcFace,    ~167 MB)
- `models/classifiers/face_detection_yunet_2023mar.onnx` (YuNet, 230 KB)
- `models/classifiers/age_net.caffemodel`     (Caffe edad, 44 MB)
- `models/classifiers/gender_net.caffemodel`  (Caffe género, 44 MB)
- `models/classifiers/opencv_face_detector_uint8.pb` (Res10 SSD, 2.6 MB)
- `models/classifiers/age_deploy.prototxt`, `gender_deploy.prototxt`,
  `opencv_face_detector.pbtxt`

Falta descargar el detector de personas YOLO11m:

```bash
python scripts/setup_models.py
```

Este script usa la API de `ultralytics` para bajar `yolo11m.pt` (~40 MB)
desde el release oficial de GitHub y lo mueve a `models/base/yolo11m.pt`.

Si quieres re-descargar/verificar también el ArcFace puedes correr:

```bash
python scripts/setup_face_embedding.py
```

## Ejecución

```bash
python main.py                       # 0.0.0.0:9000
python main.py --port 9001           # cambiar puerto
python main.py --host 127.0.0.1      # solo local
python main.py --reload              # auto-reload en dev
```

Verifica que está vivo:
```bash
curl http://localhost:9000/
curl http://localhost:9000/health
```

## Endpoint WebSocket

**URL:** `ws://<host>:9000/ws/Personal%20de%20Amazonas`

**Mensaje del cliente → servidor** (JSON o msgpack):
```jsonc
{
  "data": {
    "image": "<base64 jpeg>",          // o bytes msgpack
    "roi_coordinates": [[500,250],[900,250],[1040,560],[600,560]],
    "roi_activate": true,
    "entrega_roi_coordinates": [],     // verde (estante), opcional
    "entrega_roi_activate": false,
    "prueba1_roi_coordinates": [],     // violeta 1, opcional
    "prueba1_roi_activate": false,
    "prueba2_roi_coordinates": [],     // violeta 2, opcional
    "prueba2_roi_activate": false,
    "cosmetics_enabled": false,        // default: OFF (solo personas)
    "camera_id": 1
  }
}
```

**Respuesta del servidor → cliente:**
```jsonc
{
  "data": {
    "camera_id": 1,
    "status": "success",
    "processing_time": 0.123,
    "processed_image": "data:image/jpeg;base64,...",
    "metadata": {
      "frame_number": 42,
      "persons_inside": 3,
      "persons_in_area": 7,
      "active_tracks": 5,
      "entry_counts": {"Hombres": 4, "Mujeres": 2, "Niños": 1, "Personas": 0},
      "people_counter": { "total_unique": 7, ... },
      "demographics":   { "Hombres": 4, "Mujeres": 2, "Niños": 1, ... },
      "attendance":     { "attended_count": 0, ... },
      "seller_efficiency": { ... },
      "stock":          { ... }
    }
  }
}
```

## Estructura del repo

```
SERVER-IA/
├── main.py                                    Entry point (uvicorn)
├── requirements.txt
├── .gitignore
├── .gitattributes                             Config Git LFS
├── README.md
├── scripts/
│   ├── setup_models.py                        Descarga yolo11m.pt
│   └── setup_face_embedding.py                Descarga ArcFace
├── models/
│   ├── base/                                  YOLO (descarga con setup_models.py)
│   └── classifiers/                           ONNX + Caffe (LFS)
├── output/                                    Logs, screenshots (gitignored)
└── src/
    ├── __init__.py
    ├── analityc/
    │   ├── config/
    │   │   ├── __init__.py
    │   │   └── config.py                      Rutas + thresholds + ENV vars
    │   └── core/
    │       ├── __init__.py
    │       ├── person_amazona_inference.py    PersonAmazonas (clase ppal)
    │       ├── hardware_available.py          Deteccion CPU/GPU
    │       ├── analytics/
    │       │   ├── config.py                  AnalyticsConfig (thresholds demografia)
    │       │   ├── demographics.py            Genero/edad ensemble (YuNet+InsightFace+Caffe)
    │       │   ├── face_reidentifier.py       Embeddings ArcFace + DB persistente
    │       │   ├── people_counter.py          Conteo unico
    │       │   ├── attendance_tracker.py      Proximidad vendedor-cliente
    │       │   ├── seller_efficiency.py       Metricas + premio horario
    │       │   └── stock_monitor.py           Monitoreo de productos por ROI
    │       └── utils/
    │           ├── overlay.py                 Paneles UI sobre el frame
    │           └── logger.py                  Logs CSV + JSONL
    └── app/
        ├── __init__.py
        ├── app.py                             FastAPI + WebSocket
        └── server.py                          Wrapper uvicorn en hilo
```

## Variables de entorno

Todas las rutas/thresholds clave se pueden sobreescribir sin tocar código:

| Variable | Default | Descripción |
|---|---|---|
| `MODELS_DIR` | `<repo>/models/base` | Dónde buscar los `.pt` |
| `OUTPUT_DIR` | `<repo>/output` | Dónde guardar logs/screenshots |
| `MODEL_PATH_AMAZONAS` | `models/cosmeticos/weights/best.pt` | Modelo de productos (opcional) |
| `CONFIDENCE_THRESHOLD` | `0.12` | Confianza mínima |
| `IOU_THRESHOLD` | `0.30` | IoU para NMS |
| `INFERENCE_DEVICE` | `auto` | `cpu`, `cuda:0`, etc. |
| `SERVER_HOST` | `0.0.0.0` | Bind host |
| `SERVER_PORT` | `9000` | Puerto WebSocket |
| `CONFIG_RELOAD` | `false` | Releer ENV en cada `get_config()` |

## Modelo de cosméticos (opcional)

El procesador soporta opcionalmente detección de 16 SKUs cosméticos. Por
defecto `cosmetics_enabled=False` y NO se carga ese modelo (solo
detección de personas + demografía). El cliente puede activarlo enviando
`cosmetics_enabled: true` en el payload del WebSocket.

Si quieres usarlo, coloca tu modelo entrenado en:
```
models/cosmeticos/weights/best.pt
```
o exporta `MODEL_PATH_AMAZONAS=/ruta/a/tu/best.pt`.

## Integración con windows_managers_view

El cliente
[windows_managers_view](../windows_managers_view) se conecta automáticamente
seleccionando el layout **"Personal de Amazonas"** desde el combo de la
barra de estado. La URL del servidor se configura en el cliente; este repo
solo escucha en el puerto 9000 por defecto.

## Licencia

Uso interno ELDE — definir antes de publicar como público.
