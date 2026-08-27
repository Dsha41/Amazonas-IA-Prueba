"""
test_client_hummus.py - Cliente de prueba para el layout "Hummus".

Habla el MISMO protocolo que Amazonasview: se conecta a
ws://host:9000/ws/<layout>, espera el connection_init, y manda frames en
el sobre {event, id_connection, type_inference, component_key, data}.

Sirve para comprobar el circuito completo del servidor sin necesidad de
levantar la aplicacion de escritorio (que requiere Windows para la
captura de ventana del DVR).

Comprueba tres cosas:
  1. Que el servidor ACEPTA "Hummus" en vez de cerrar con codigo 1008.
  2. Que devuelve la imagen procesada y la metadata.
  3. Que los eventos de mostrador llegan en metadata["alerts"] con el
     formato que el cliente real espera (render_box.py lee event_type,
     class_name, description, timestamp, image_base64).

Uso:
    python test_client_hummus.py                      # layout Hummus
    python test_client_hummus.py --layout "Personal de Amazonas"
    python test_client_hummus.py --layout NoExiste    # debe rechazarlo
"""

import argparse
import asyncio
import base64
import json
import sys
import uuid
from urllib.parse import quote

import cv2
import websockets

VIDEO = "toma de orden y entrega de plato.avi"


def codificar(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("no se pudo codificar el frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost:9000")
    ap.add_argument("--layout", default="Hummus")
    ap.add_argument("--desde", type=int, default=0, help="frame inicial")
    ap.add_argument("--frames", type=int, default=60, help="cuantos enviar")
    ap.add_argument("--salto", type=int, default=1, help="1 de cada N")
    args = ap.parse_args()

    # El nombre del layout va en la ruta y puede llevar espacios
    # ("Personal de Amazonas"), asi que hay que codificarlo.
    url = f"ws://{args.host}/ws/{quote(args.layout)}"
    print(f"Conectando a {url}")

    try:
        ws = await websockets.connect(url, max_size=64 * 1024 * 1024)
    except Exception as e:
        print(f"  no se pudo conectar: {e}")
        return 1

    async with ws:
        # ── Handshake ────────────────────────────────────────────────
        try:
            saludo = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        except Exception as e:
            print(f"  sin respuesta al conectar: {e}")
            return 1

        if saludo.get("status") == "error":
            print(f"  RECHAZADO: {saludo.get('message')}")
            return 2

        if saludo.get("event") != "connection_init":
            print(f"  respuesta inesperada: {saludo}")
            return 1

        id_conn = saludo.get("id_connection")
        print(f"  ACEPTADO. id_connection={id_conn}, "
              f"type_inference={saludo.get('type_inference')!r}")

        # ── Envio de frames ──────────────────────────────────────────
        cap = cv2.VideoCapture(VIDEO)
        if not cap.isOpened():
            print(f"  no se pudo abrir {VIDEO}")
            return 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.desde)

        component_key = str(uuid.uuid4())
        alertas_totales = []
        enviados = 0
        idx = args.desde

        print(f"\n  enviando frames desde {args.desde} "
              f"(1 de cada {args.salto})...")
        print(f"  {'frame':>6} {'personas':>9} {'alertas':>8}  evento")

        while enviados < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if (idx - 1 - args.desde) % args.salto != 0:
                continue

            sobre = {
                "event": "inference",
                "id_connection": id_conn,
                "type_inference": args.layout,
                "component_key": component_key,
                "data": {
                    "image": codificar(frame),
                    "camera_id": component_key,
                    "roi_activate": False,
                    "cosmetics_enabled": False,
                },
            }
            await ws.send(json.dumps(sobre))

            # El servidor manda pings mientras procesa; se ignoran.
            while True:
                try:
                    crudo = await asyncio.wait_for(ws.recv(), timeout=120)
                except asyncio.TimeoutError:
                    print("  sin respuesta en 120s")
                    return 1
                msg = json.loads(crudo)
                if msg.get("status") == "ping":
                    continue
                break

            enviados += 1
            data = msg.get("data", {}) or {}
            if data.get("status") == "error":
                print(f"  ERROR del servidor: {data.get('message')}")
                return 1

            meta = data.get("metadata", {}) or {}
            alertas = meta.get("alerts", []) or []
            alertas_totales.extend(alertas)

            marca = ""
            if alertas:
                marca = "  <-- " + ", ".join(a.get("event_type", "?")
                                             for a in alertas)
            if alertas or enviados % 10 == 0:
                print(f"  {idx:>6} {meta.get('persons_inside', 0):>9} "
                      f"{len(alertas_totales):>8}{marca}")

        cap.release()

    # ── Informe ──────────────────────────────────────────────────────
    print(f"\n  {enviados} frames enviados, "
          f"{len(alertas_totales)} alerta(s) recibida(s)")

    if not alertas_totales:
        print("  (ninguna alerta; puede ser normal segun el tramo enviado)")
        return 0

    print("\n  === ALERTAS RECIBIDAS ===")
    campos_cliente = ["event_type", "class_name", "description",
                      "timestamp", "image_base64"]
    for a in alertas_totales:
        print(f"\n  event_type : {a.get('event_type')!r}")
        print(f"  class_name : {a.get('class_name')}")
        print(f"  description: {a.get('description')}")
        print(f"  timestamp  : {a.get('timestamp')}")
        img = a.get("image_base64", "")
        print(f"  imagen     : {len(img)} caracteres base64")
        faltan = [c for c in campos_cliente if not a.get(c)]
        if faltan:
            print(f"  FALTAN campos que el cliente usa: {faltan}")
        else:
            print("  todos los campos que lee render_box.py estan presentes")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
