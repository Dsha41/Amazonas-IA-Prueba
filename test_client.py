"""
test_client.py - Cliente de prueba para SERVER-IA Amazonas.

Extrae un frame de un video y lo envia al servidor por WebSocket
para verificar que el pipeline completo (YOLO + demografia + Re-ID)
responde correctamente.

Uso:
    python test_client.py                          # usa el video por defecto
    python test_client.py --video otro.mp4         # otro video
    python test_client.py --frame 100              # frame especifico
"""

import argparse
import asyncio
import base64
import json
import sys

import cv2
import websockets

DEFAULT_VIDEO = "159458-818908448.mp4"
SERVER_URL = "ws://localhost:9000/ws/Personal%20de%20Amazonas"


def extraer_frame(video_path: str, n_frame: int):
    """Lee el frame n del video y lo devuelve como JPEG en base64."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] No se pudo abrir el video: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Video abierto: {total} frames totales")

    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"[ERR] No se pudo leer el frame {n_frame}")
        sys.exit(1)

    alto, ancho = frame.shape[:2]
    print(f"[*] Frame {n_frame} extraido: {ancho}x{alto}")

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        print("[ERR] No se pudo codificar el frame a JPEG")
        sys.exit(1)

    return base64.b64encode(buffer).decode("utf-8")


async def enviar(imagen_b64: str, url: str):
    """Conecta al servidor, envia el frame y muestra la respuesta."""
    print(f"[*] Conectando a {url}")

    async with websockets.connect(url, max_size=50 * 1024 * 1024) as ws:
        print("[OK] Conectado")

        mensaje = {
            "data": {
                "image": imagen_b64,
                "roi_activate": False,
                "cosmetics_enabled": False,
                "camera_id": 1,
            }
        }

        print("[*] Enviando frame... (la primera vez tarda: carga los modelos)")
        await ws.send(json.dumps(mensaje))

        while True:
            respuesta = await asyncio.wait_for(ws.recv(), timeout=300)

            if isinstance(respuesta, bytes):
                print("[!] Respuesta binaria (msgpack), no esperada con JSON")
                break

            data = json.loads(respuesta)

            # Handshake inicial: lo ignoramos
            if data.get("event") == "connection_init":
                print("    ...handshake recibido, conexion establecida")
                continue

            # El servidor manda pings mientras procesa
            if data.get("status") == "ping":
                print("    ...ping (servidor sigue trabajando)")
                continue

            mostrar_respuesta(data)
            break


def mostrar_respuesta(data: dict):
    """Imprime la respuesta de forma legible."""
    print("\n" + "=" * 60)
    print("RESPUESTA DEL SERVIDOR")
    print("=" * 60)

    payload = data.get("data", data)

    status = payload.get("status", "?")
    print(f"status          : {status}")

    if "processing_time" in payload:
        print(f"processing_time : {payload['processing_time']:.3f} s")

    if "error" in payload or status == "error":
        print(f"error           : {payload.get('error') or payload.get('message')}")
        print("=" * 60)
        return

    # La imagen procesada es enorme en base64: solo confirmamos que vino
    img = payload.get("processed_image")
    if img:
        print(f"processed_image : recibida ({len(img):,} chars base64)")
        guardar_imagen(img)

    metadata = payload.get("metadata")
    if metadata:
        print("\nMETADATA (analitica):")
        print(json.dumps(metadata, indent=2, ensure_ascii=False)[:3000])

    print("=" * 60)


def guardar_imagen(img_b64: str):
    """Guarda la imagen anotada para poder verla."""
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    try:
        with open("resultado.jpg", "wb") as f:
            f.write(base64.b64decode(img_b64))
        print("                  -> guardada como resultado.jpg")
    except Exception as exc:
        print(f"                  -> no se pudo guardar: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--url", default=SERVER_URL)
    args = parser.parse_args()

    imagen = extraer_frame(args.video, args.frame)
    asyncio.run(enviar(imagen, args.url))


if __name__ == "__main__":
    main()