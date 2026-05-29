"""
app.py - Servidor FastAPI + WebSocket para PersonAmazonas.

Version simplificada que SOLO soporta el procesador PersonAmazonas
(deteccion de personas + demografia + face re-identification + analitica
retail). El endpoint WebSocket /ws/Personal%20de%20Amazonas recibe frames
del cliente Amazonas View (windows_managers_view) y devuelve la imagen
procesada con metadata.

Protocolo de mensajes:
    Cliente -> Servidor:
        {
            "data": {
                "image": "<base64 jpeg | bytes>",
                "roi_coordinates": [[x,y], ...],
                "roi_activate": true|false,
                "entrega_roi_coordinates": [[x,y], ...],   (verde, opcional)
                "entrega_roi_activate": true|false,
                "prueba1_roi_coordinates": [[x,y], ...],   (violeta 1, opcional)
                "prueba1_roi_activate": true|false,
                "prueba2_roi_coordinates": [[x,y], ...],   (violeta 2, opcional)
                "prueba2_roi_activate": true|false,
                "cosmetics_enabled": true|false,
                "camera_id": 1
            }
        }

    Servidor -> Cliente:
        {
            "data": {
                "camera_id": 1,
                "status": "success",
                "metadata": { ... analitica ... },
                "processed_image": "data:image/jpeg;base64,...",
                "processing_time": 0.123
            }
        }
"""

import asyncio
import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple

import cv2
import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

from ..analityc.core.person_amazona_inference import PersonAmazonas
from ..analityc.core.hardware_available import device_hardware
from ..analityc.config.config import get_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
WEBSOCKET_TIMEOUT: float = 30.0      # segundos sin mensaje antes de ping
MAX_QUEUE_PER_CLIENT: int = 2        # frames maximos en cola por cliente
JPEG_QUALITY: int = 70
EXECUTOR_WORKERS: int = 8

SUPPORTED_INFERENCE = "Personal de Amazonas"


# -----------------------------------------------------------------------------
# ThreadPoolExecutor + lifespan
# -----------------------------------------------------------------------------
executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)


def _get_live_executor() -> ThreadPoolExecutor:
    global executor
    if getattr(executor, "_shutdown", False):
        logger.info("Recreando ThreadPoolExecutor")
        executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
    return executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    if getattr(executor, "_shutdown", False):
        executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
    logger.info("Servidor iniciando - ThreadPoolExecutor con %d workers",
                EXECUTOR_WORKERS)
    yield
    logger.info("Servidor apagandose - cerrando ThreadPoolExecutor")
    executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan, title="SERVER-IA Amazonas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Estado global de conexiones
# -----------------------------------------------------------------------------
active_connections: Dict[str, Dict[str, Any]] = {}
connection_lock = asyncio.Lock()


# -----------------------------------------------------------------------------
# Helpers de hardware / configuracion
# -----------------------------------------------------------------------------

def _get_first_gpu() -> str:
    gpus = getattr(device_hardware, "gpu_tuple", [])
    if isinstance(gpus, (list, tuple)) and gpus:
        gpu0 = gpus[0]
        if isinstance(gpu0, dict):
            return gpu0.get("gpu_use", "cpu")
    dev = getattr(device_hardware, "device_default", "cpu")
    if isinstance(dev, dict):
        return dev.get("gpu_use", "cpu")
    return dev if isinstance(dev, str) else "cpu"


def _build_processor(client_id: str, config: Dict[str, Any]) -> PersonAmazonas:
    """Instancia el procesador PersonAmazonas."""
    gpu = _get_first_gpu()
    model_paths = config.get("model_paths", {}) or {}
    person_paths = config.get("person_model_paths", {}) or {}

    return PersonAmazonas(
        client_id=client_id,
        model_path=model_paths.get("Personal de Amazonas"),
        person_model_path=person_paths.get("Personal de Amazonas"),
        confidence_threshold=config.get("confidence_threshold", 0.5),
        iou_threshold=config.get("iou_threshold", 0.4),
        device=gpu,
    )


# -----------------------------------------------------------------------------
# Procesamiento de frames
# -----------------------------------------------------------------------------

def _decode_image(image_data: Any) -> np.ndarray:
    """Decodifica desde base64-str, bytes o data-URI."""
    if isinstance(image_data, (bytes, bytearray)):
        image_bytes = bytes(image_data)
    elif isinstance(image_data, str):
        if image_data.startswith("data:") and "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
    else:
        raise ValueError(f"Tipo de imagen no soportado: {type(image_data)}")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen corrupta o formato invalido")
    return img


def _encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Error en la codificacion JPEG de salida")
    return buf.tobytes()


def _process_amazonas(
    processor: PersonAmazonas,
    img: np.ndarray,
    roi: Any,
    roi_activate: bool,
    entrega_roi: Any,
    entrega_roi_activate: bool,
    prueba1_roi: Any,
    prueba1_roi_activate: bool,
    prueba2_roi: Any,
    prueba2_roi_activate: bool,
    cosmetics_enabled: bool,
    camera_id: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Procesa el frame con PersonAmazonas y devuelve (imagen, metadata)."""
    esc_roi = entrega_roi if (entrega_roi_activate
                              and isinstance(entrega_roi, list)
                              and len(entrega_roi) >= 3) else None
    p1_roi = prueba1_roi if (prueba1_roi_activate
                             and isinstance(prueba1_roi, list)
                             and len(prueba1_roi) >= 3) else None
    p2_roi = prueba2_roi if (prueba2_roi_activate
                             and isinstance(prueba2_roi, list)
                             and len(prueba2_roi) >= 3) else None

    try:
        cam_proc = processor.get_camera_processor(camera_id)
        cam_proc.cosmetics_enabled = bool(cosmetics_enabled)
        return cam_proc.process_frame(
            img, roi, roi_activate, camera_id,
            roi_escaparate=esc_roi,
            roi_prueba1=p1_roi,
            roi_prueba2=p2_roi,
        )
    except Exception:
        processor.cosmetics_enabled = bool(cosmetics_enabled)
        return processor.process_frame(
            img, roi, roi_activate, camera_id,
            roi_escaparate=esc_roi,
            roi_prueba1=p1_roi,
            roi_prueba2=p2_roi,
        )


def _full_frame_sync(
    processor: PersonAmazonas,
    image_data: Any,
    roi: Any,
    roi_activate: bool,
    camera_id: Any,
    entrega_roi: Any,
    entrega_roi_activate: bool,
    prueba1_roi: Any,
    prueba1_roi_activate: bool,
    prueba2_roi: Any,
    prueba2_roi_activate: bool,
    cosmetics_enabled: bool,
) -> Dict[str, Any]:
    """Decode + inferencia + encode JPEG. Ejecutado en el ThreadPoolExecutor."""
    t0 = time.time()
    img = _decode_image(image_data)
    processed_img, metadata = _process_amazonas(
        processor, img, roi, roi_activate,
        entrega_roi, entrega_roi_activate,
        prueba1_roi, prueba1_roi_activate,
        prueba2_roi, prueba2_roi_activate,
        cosmetics_enabled, camera_id,
    )
    jpeg_bytes = _encode_jpeg(processed_img)
    return {
        "camera_id": camera_id,
        "status": "success",
        "metadata": metadata,
        "processed_image": f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode()}",
        "processing_time": round(time.time() - t0, 3),
    }


# -----------------------------------------------------------------------------
# Worker por cliente
# -----------------------------------------------------------------------------

class ClientWorker:
    """Cola y worker dedicados a UN cliente (evita head-of-line blocking)."""

    def __init__(self, client_id: str, processor: PersonAmazonas):
        self.client_id = client_id
        self.processor = processor
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_PER_CLIENT)
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    def start(self) -> None:
        self._task = asyncio.create_task(
            self._run(), name=f"worker-{self.client_id}"
        )

    async def stop(self) -> None:
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        while not self._queue.empty():
            try:
                fut, _ = self._queue.get_nowait()
                if not fut.done():
                    fut.cancel()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def enqueue(self, payload: tuple) -> asyncio.Future:
        """Drop-head: si la cola esta llena, descarta el frame mas viejo."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        if self._queue.full():
            try:
                old_fut, _ = self._queue.get_nowait()
                self._queue.task_done()
                if not old_fut.done():
                    old_fut.set_result(
                        {"status": "dropped", "reason": "queue_full"}
                    )
            except asyncio.QueueEmpty:
                pass

        await self._queue.put((future, payload))
        return future

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        while not self._stopped:
            try:
                future, payload = await self._queue.get()
            except asyncio.CancelledError:
                break

            if future.done():
                self._queue.task_done()
                continue

            try:
                result = await loop.run_in_executor(
                    _get_live_executor(), _full_frame_sync, *payload
                )
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                logger.error("Error en worker client=%s: %s", self.client_id, exc)
                if not future.done():
                    future.set_result({"status": "error", "message": str(exc)})
            finally:
                self._queue.task_done()


# -----------------------------------------------------------------------------
# WebSocket endpoint
# -----------------------------------------------------------------------------

@app.websocket("/ws/{type_inference}")
async def websocket_endpoint(websocket: WebSocket, type_inference: str):
    await websocket.accept()
    client_id = f"client_{id(websocket)}"
    worker: Optional[ClientWorker] = None
    processor: Optional[PersonAmazonas] = None

    if type_inference != SUPPORTED_INFERENCE:
        await websocket.send_text(json.dumps({
            "status": "error",
            "message": (f"Tipo de inferencia '{type_inference}' no soportado. "
                        f"Este servidor solo expone '{SUPPORTED_INFERENCE}'.")
        }))
        await websocket.close(code=1008)
        return

    try:
        config = get_config()
        processor = _build_processor(client_id, config)

        worker = ClientWorker(client_id, processor)
        worker.start()

        async with connection_lock:
            active_connections[client_id] = {
                "websocket": websocket,
                "processor": processor,
                "worker": worker,
                "type_inference": type_inference,
                "connected_at": time.time(),
                "last_active": time.time(),
            }

        logger.info("Cliente conectado: %s", client_id)

        await websocket.send_text(json.dumps({
            "id_connection": id(websocket),
            "event": "connection_init",
            "type_inference": type_inference,
            "data": {"roi": False},
        }))

        # -- Bucle principal -------------------------------------------------
        while True:
            try:
                recv = await asyncio.wait_for(
                    websocket.receive(), timeout=WEBSOCKET_TIMEOUT
                )
            except asyncio.TimeoutError:
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(json.dumps({"status": "ping"}))
                    except Exception:
                        break
                continue

            incoming_is_binary = recv.get("bytes") is not None
            try:
                if incoming_is_binary:
                    request = msgpack.unpackb(
                        recv["bytes"], raw=False, strict_map_key=False
                    )
                else:
                    request = json.loads(recv.get("text") or "")
            except Exception as exc:
                await _send_error(websocket, incoming_is_binary, str(exc))
                continue

            data = request.get("data", {})
            if not isinstance(data, dict) or "image" not in data:
                await _send_error(
                    websocket, incoming_is_binary,
                    "Campo 'image' requerido dentro de 'data'"
                )
                continue

            image_data = data["image"]
            roi = data.get("roi_coordinates", "")
            roi_activate = bool(data.get("roi_activate", False))
            entrega_roi = data.get("entrega_roi_coordinates")
            entrega_roi_activate = bool(data.get("entrega_roi_activate", False))
            prueba1_roi = data.get("prueba1_roi_coordinates")
            prueba1_roi_activate = bool(data.get("prueba1_roi_activate", False))
            prueba2_roi = data.get("prueba2_roi_coordinates")
            prueba2_roi_activate = bool(data.get("prueba2_roi_activate", False))
            cosmetics_enabled = bool(data.get("cosmetics_enabled", True))

            camera_id_raw = data.get("camera_id", 1)
            try:
                camera_id = int(camera_id_raw)
            except (TypeError, ValueError):
                camera_id = camera_id_raw

            frame_payload = (
                processor, image_data,
                roi, roi_activate,
                camera_id,
                entrega_roi, entrega_roi_activate,
                prueba1_roi, prueba1_roi_activate,
                prueba2_roi, prueba2_roi_activate,
                cosmetics_enabled,
            )

            try:
                future = await worker.enqueue(frame_payload)
                result = await future
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error encolando frame: %s", exc)
                result = {"status": "error", "message": str(exc)}

            if websocket.client_state != WebSocketState.CONNECTED:
                break

            request["data"] = result
            try:
                if incoming_is_binary:
                    await websocket.send_bytes(
                        msgpack.packb(request, use_bin_type=True)
                    )
                else:
                    await websocket.send_json(request)
            except Exception as exc:
                logger.error("Error enviando respuesta: %s", exc)
                break

            async with connection_lock:
                if client_id in active_connections:
                    active_connections[client_id]["last_active"] = time.time()

    except WebSocketDisconnect:
        logger.info("Cliente desconectado: %s", client_id)
    except Exception as exc:
        logger.error("Error critico client=%s: %s", client_id, exc)
    finally:
        if worker:
            await worker.stop()
        async with connection_lock:
            active_connections.pop(client_id, None)
        if processor and hasattr(processor, "cleanup"):
            try:
                processor.cleanup()
            except Exception:
                pass
        logger.info("Limpieza completa client=%s", client_id)


async def _send_error(websocket: WebSocket, is_binary: bool, message: str) -> None:
    payload = {"data": {"status": "error", "message": message}}
    try:
        if is_binary:
            await websocket.send_bytes(msgpack.packb(payload, use_bin_type=True))
        else:
            await websocket.send_json(payload)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Endpoints HTTP
# -----------------------------------------------------------------------------

@app.get("/")
def init_server():
    return {
        "status": "active",
        "service": "SERVER-IA Amazonas",
        "connections": len(active_connections),
        "supported_inference": SUPPORTED_INFERENCE,
    }


@app.get("/health")
def health_check():
    now = time.time()
    clients = [
        {
            "client_id": cid,
            "type_inference": info.get("type_inference", "unknown"),
            "idle_seconds": round(now - info.get("last_active", now), 1),
            "connected_seconds": round(now - info.get("connected_at", now), 1),
        }
        for cid, info in active_connections.items()
    ]
    return {
        "status": "active",
        "total_connections": len(active_connections),
        "executor_max_workers": EXECUTOR_WORKERS,
        "clients": clients,
    }
