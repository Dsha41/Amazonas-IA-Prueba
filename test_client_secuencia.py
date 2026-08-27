"""
test_client_secuencia.py - Envia varios frames consecutivos al servidor.

A diferencia de test_client.py (un solo frame), este script manda una
secuencia por la MISMA conexion. Esto permite verificar:

  - El voting temporal de demografia: el sistema acumula muestras de cada
    persona antes de clasificar genero/edad. Con un solo frame todo queda
    "pending"; con varios, deberian empezar a clasificarse.
  - El tracking: los IDs de las personas deben mantenerse entre frames.
  - El rendimiento real: el primer frame incluye la carga de modelos,
    los siguientes no.

Uso:
    python test_client_secuencia.py
    python test_client_secuencia.py --inicio 50 --cantidad 30 --salto 2
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


def extraer_frames(video_path: str, inicio: int, cantidad: int, salto: int):
    """
    Extrae una lista de frames del video, ya codificados en base64.

    salto=1 -> frames consecutivos (50, 51, 52...)
    salto=2 -> uno de cada dos (50, 52, 54...) para cubrir mas tiempo
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] No se pudo abrir el video: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"[*] Video: {total} frames a {fps:.0f} fps")

    frames = []
    numeros = []

    for i in range(cantidad):
        n = inicio + (i * salto)
        if n >= total:
            print(f"[!] Frame {n} fuera de rango, corto aqui")
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        if not ok:
            print(f"[!] No se pudo leer el frame {n}, lo salto")
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frames.append(base64.b64encode(buffer).decode("utf-8"))
        numeros.append(n)

    cap.release()

    if not frames:
        print("[ERR] No se extrajo ningun frame")
        sys.exit(1)

    span = (numeros[-1] - numeros[0]) / fps
    print(f"[*] {len(frames)} frames extraidos "
          f"({numeros[0]} a {numeros[-1]}, ~{span:.1f}s de video)")
    return frames, numeros


async def esperar_respuesta(ws, timeout=300):
    """
    Espera la respuesta real del servidor, ignorando handshake y pings.
    Devuelve el diccionario de la respuesta.
    """
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)

        if isinstance(msg, bytes):
            raise RuntimeError("Respuesta binaria inesperada")

        data = json.loads(msg)

        if data.get("event") == "connection_init":
            continue
        if data.get("status") == "ping":
            continue

        return data


async def enviar_secuencia(frames, numeros, url):
    print(f"\n[*] Conectando a {url}")

    async with websockets.connect(url, max_size=50 * 1024 * 1024) as ws:
        print("[OK] Conectado\n")

        print(f"{'frame':>6} {'tiempo':>8} {'tracks':>7} {'unicos':>7} "
              f"{'clasif':>7} {'pend':>6} {'unk':>5}")
        print("-" * 52)

        ultimo_payload = None

        for i, (img, n) in enumerate(zip(frames, numeros)):
            mensaje = {
                "data": {
                    "image": img,
                    "roi_activate": False,
                    "cosmetics_enabled": False,
                    "camera_id": 1,
                }
            }

            await ws.send(json.dumps(mensaje))
            data = await esperar_respuesta(ws)
            payload = data.get("data", data)

            if payload.get("status") == "error":
                print(f"\n[ERR] frame {n}: "
                      f"{payload.get('error') or payload.get('message')}")
                break

            ultimo_payload = payload
            meta = payload.get("metadata", {})
            demo = meta.get("demographics", {})
            counter = meta.get("people_counter", {})

            print(f"{n:>6} "
                  f"{payload.get('processing_time', 0):>7.3f}s "
                  f"{meta.get('active_tracks', 0):>7} "
                  f"{counter.get('total_unique', 0):>7} "
                  f"{demo.get('total_classified', 0):>7} "
                  f"{demo.get('total_pending', 0):>6} "
                  f"{demo.get('total_unknown', 0):>5}")

            if i == 0:
                print("       ^ primer frame incluye carga de modelos")

        if ultimo_payload:
            resumen_final(ultimo_payload)


def resumen_final(payload):
    """Muestra la metadata completa del ultimo frame procesado."""
    print("\n" + "=" * 60)
    print("ESTADO FINAL (ultimo frame)")
    print("=" * 60)

    meta = payload.get("metadata", {})

    demo = meta.get("demographics", {})
    print("\nDEMOGRAFIA:")
    print(f"  clasificados : {demo.get('total_classified', 0)}")
    print(f"  pendientes   : {demo.get('total_pending', 0)}")
    print(f"  desconocidos : {demo.get('total_unknown', 0)}")

    if demo.get("gender"):
        print(f"  genero       : {demo['gender']}")
    if demo.get("age"):
        print(f"  edad         : {demo['age']}")
    if demo.get("combined"):
        print(f"  combinado    : {demo['combined']}")

    print("\nCONTEO:")
    counter = meta.get("people_counter", {})
    print(f"  unicos totales : {counter.get('total_unique', 0)}")
    print(f"  activos ahora  : {counter.get('active_now', 0)}")
    print(f"  por categoria  : {meta.get('active_by_category', {})}")

    print("\nATENCION:")
    att = meta.get("attendance", {})
    print(f"  atendidos    : {att.get('attended_count', 0)}")
    print(f"  proximidades : {att.get('active_proximities', 0)}")
    print(f"  vendedores   : {att.get('seller_count', 0)}")

    img = payload.get("processed_image")
    if img:
        guardar_imagen(img)

    print("=" * 60)


def guardar_imagen(img_b64: str, nombre="resultado_final.jpg"):
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    try:
        with open(nombre, "wb") as f:
            f.write(base64.b64decode(img_b64))
        print(f"\n[OK] Ultimo frame anotado guardado como {nombre}")
    except Exception as exc:
        print(f"\n[!] No se pudo guardar la imagen: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--inicio", type=int, default=50,
                        help="numero del primer frame")
    parser.add_argument("--cantidad", type=int, default=30,
                        help="cuantos frames enviar")
    parser.add_argument("--salto", type=int, default=2,
                        help="1=consecutivos, 2=uno de cada dos, etc.")
    parser.add_argument("--url", default=SERVER_URL)
    args = parser.parse_args()

    frames, numeros = extraer_frames(
        args.video, args.inicio, args.cantidad, args.salto
    )
    asyncio.run(enviar_secuencia(frames, numeros, args.url))


if __name__ == "__main__":
    main()