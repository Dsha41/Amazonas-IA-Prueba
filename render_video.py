"""
render_video.py - Genera el video anotado a partir de los tracks ya
volcados, SIN volver a correr YOLO.

Dibuja sobre el video original: las zonas, cada persona con su
identificador y el rol asignado, la distancia al empleado mas cercano, y
un registro de los eventos segun se van confirmando. Sirve para juzgar a
ojo si los eventos caen donde deben, que es algo que el conteo no dice.

Uso:
    python render_video.py [--escala 0.75] [--salida output/hummus_anotado.webm]
"""

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

from src.analityc.core.analytics.zone_event_tracker import ZoneEventTracker
from src.analityc.core.analytics.zone_dwell_tracker import ZoneDwellTracker
from test_hummus_zones import load_zones

VIDEO_PATH = "toma de orden y entrega de plato.avi"

# Los parametros ya NO viven aqui: se leen de hummus_zones.json, que es
# la unica fuente de verdad. Tener el mismo numero repartido en tres
# archivos es exactamente como acaban desincronizados.

C_CAJA = (60, 200, 230)
C_ENTREGA = (40, 140, 240)
C_STAFF = (200, 80, 200)
C_FUERA = (140, 140, 140)
C_LINEA = (255, 255, 255)


def cargar_todos():
    """Junta todos los volcados de tracks disponibles, ordenados."""
    porframe = {}
    fps = 11.919
    for path in sorted(glob.glob("output/hummus_tracks_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        fps = data.get("fps", fps)
        for fr in data["frames"]:
            porframe[fr["frame"]] = {
                "t": fr["t"],
                "tracks": {int(tid): {"center": tuple(t["center"]), "box": t["box"]}
                           for tid, t in fr["tracks"].items()},
            }
    return fps, porframe


def texto(img, s, org, color, escala=0.5, grosor=1):
    """Texto con contorno oscuro para que se lea sobre cualquier fondo."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escala", type=float, default=0.75,
                    help="Reescalado de salida (0.75 = 720x432)")
    ap.add_argument("--salto", type=int, default=1,
                    help="Escribir 1 de cada N frames. El video mantiene su "
                         "duracion real (baja el fps de salida), solo pierde "
                         "fluidez. Sirve para bajar el peso del archivo.")
    ap.add_argument("--salida", default="output/hummus_anotado.webm")
    args = ap.parse_args()

    cfg = load_zones()
    fps, porframe = cargar_todos()
    if not porframe:
        raise SystemExit("No hay volcados de tracks. Corre test_hummus_zones.py primero.")

    zc = np.array(cfg["zona_caja"], np.int32).reshape((-1, 1, 2))
    ze = np.array(cfg["zona_entrega"], np.int32).reshape((-1, 1, 2))
    zs = np.array(cfg["lado_staff"], np.int32).reshape((-1, 1, 2))

    entrega = ZoneEventTracker(
        event_name="ENTREGA", zone_polygon=cfg["zona_entrega"],
        staff_side_polygon=cfg["lado_staff"],
        distance_px=cfg["distance_px"], min_time_sec=cfg["min_time_sec"],
        max_gap_sec=cfg["max_gap_sec"],
        staff_pair_polygon=cfg.get("zona_servidor"),
    )
    orden = ZoneDwellTracker(
        event_name="ORDEN", zone_polygon=cfg["zona_caja"],
        staff_side_polygon=cfg["lado_staff"],
        min_dwell_sec=cfg["orden_dwell_sec"],
        max_gap_sec=cfg["orden_gap_sec"],
    )
    print("  entrega: {:.0f}px / {:.1f}s (tolerancia {:.1f}s)".format(
        cfg["distance_px"], cfg["min_time_sec"], cfg["max_gap_sec"]))
    print("  orden:   permanencia {:.1f}s".format(cfg["orden_dwell_sec"]))

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {VIDEO_PATH}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    outW, outH = int(W * args.escala), int(H * args.escala)

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.salida, cv2.VideoWriter_fourcc(*"VP80"),
                         fps / args.salto, (outW, outH))
    if not vw.isOpened():
        raise SystemExit("No se pudo abrir el codificador VP80")

    # El texto se dibuja sobre el frame a tamano original y luego se
    # reduce, asi que hay que agrandarlo para que siga legible. La
    # compensacion va amortiguada (0.6): compensar la escala entera
    # deja las etiquetas enormes y solapadas sobre una escena con
    # ocho personas.
    k = 1.0 + (1.0 / max(args.escala, 0.1) - 1.0) * 0.6

    registro = []          # eventos confirmados, para el panel inferior
    destellos = {}         # frame_idx -> texto, para el aviso grande
    frames_con_tracks = 0

    print(f"Renderizando {total} frames a {outW}x{outH} ...")

    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break

        datos = porframe.get(idx)
        overlay = frame.copy()
        cv2.polylines(overlay, [zs], True, C_STAFF, 2)
        cv2.polylines(overlay, [zc], True, C_CAJA, 2)
        cv2.polylines(overlay, [ze], True, C_ENTREGA, 2)
        texto(overlay, "LADO STAFF", tuple(cfg["lado_staff"][0]), C_STAFF, 0.5 * k, 2)
        texto(overlay, "CAJA", tuple(cfg["zona_caja"][0]), C_CAJA, 0.5 * k, 2)
        texto(overlay, "ENTREGA", tuple(cfg["zona_entrega"][0]), C_ENTREGA, 0.5 * k, 2)

        if datos is None:
            texto(overlay, "(sin deteccion en este tramo)", (12, H - 14), C_FUERA, 0.6 * k, 2)
        else:
            frames_con_tracks += 1
            t = datos["t"]
            tracks = datos["tracks"]

            for e in entrega.update(tracks, now=t):
                linea = (f"t={e['start_time']:.1f}s  ENTREGA  "
                         f"empleado {e['staff_id']} -> cliente {e['client_id']}")
                registro.append((linea, C_ENTREGA))
                destellos[idx] = ("ENTREGA DE PLATO", C_ENTREGA)
            for e in orden.update(tracks, now=t):
                linea = (f"t={e['start_time']:.1f}s  ORDEN     "
                         f"cliente {e['client_id']} ({e['duration']:.0f}s en caja)")
                registro.append((linea, C_CAJA))
                destellos[idx] = ("TOMA DE ORDEN", C_CAJA)

            staff, cli_e = [], []
            for tid, tr in tracks.items():
                c = tr["center"]
                cx, cy = int(c[0]), int(c[1])
                if entrega._is_staff(c):
                    rol, col = "STAFF", C_STAFF
                    staff.append((tid, c))
                elif entrega._is_inside(c, ze):
                    rol, col = "CLI-ENTREGA", C_ENTREGA
                    cli_e.append((tid, c))
                elif entrega._is_inside(c, zc):
                    rol, col = "CLI-CAJA", C_CAJA
                else:
                    rol, col = "", C_FUERA

                if tr["box"]:
                    x1, y1, x2, y2 = [int(v) for v in tr["box"]]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
                cv2.drawMarker(overlay, (cx, cy), col, cv2.MARKER_CROSS, 12, 2)
                etiqueta = f"{tid}" + (f" {rol}" if rol else "")
                texto(overlay, etiqueta, (cx - 34, cy - 12), col, 0.45 * k, 2)

            # Distancia del par mas cercano en la zona de entrega
            if staff and cli_e:
                d, sc, cc = min(
                    (math.hypot(sc[0] - cc[0], sc[1] - cc[1]), sc, cc)
                    for _, sc in staff for _, cc in cli_e)
                p1 = (int(sc[0]), int(sc[1]))
                p2 = (int(cc[0]), int(cc[1]))
                cerca = d <= cfg["distance_px"]
                cv2.line(overlay, p1, p2, C_LINEA if cerca else C_FUERA,
                         2 if cerca else 1, cv2.LINE_AA)
                texto(overlay, f"{d:.0f}px",
                      ((p1[0] + p2[0]) // 2 - 26, (p1[1] + p2[1]) // 2 - 8),
                      C_LINEA if cerca else C_FUERA, 0.6 * k, 2)

            texto(overlay, f"t={t:5.1f}s   frame {idx}   personas: {len(tracks)}",
                  (12, int(26 * k)), (255, 255, 255), 0.6 * k, 2)

        # Aviso grande durante ~1.5s tras confirmar un evento
        for fidx, (msg, col) in list(destellos.items()):
            if 0 <= idx - fidx <= int(fps * 1.5):
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                cv2.rectangle(overlay, (W // 2 - tw // 2 - 16, 44),
                              (W // 2 + tw // 2 + 16, 44 + th + 22), col, -1)
                cv2.putText(overlay, msg, (W // 2 - tw // 2, 44 + th + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
            elif idx - fidx > int(fps * 1.5):
                del destellos[fidx]

        # Panel con los ultimos eventos confirmados
        if registro:
            ultimos = registro[-4:]
            paso = int(22 * k)
            y = H - 12 - paso * (len(ultimos) - 1)
            cv2.rectangle(overlay, (0, y - paso), (W, H), (0, 0, 0), -1)
            for linea, col in ultimos:
                texto(overlay, linea, (12, y), col, 0.52 * k, 2)
                y += paso

        if idx % args.salto == 0:
            vw.write(cv2.resize(overlay, (outW, outH), interpolation=cv2.INTER_AREA))

        if idx % 200 == 0:
            print(f"  {idx}/{total}")

    cap.release()
    vw.release()

    mb = os.path.getsize(args.salida) / 1024 / 1024
    print(f"\nListo: {args.salida}  ({mb:.2f} MB)")
    print(f"  frames con detecciones: {frames_con_tracks}/{total}")
    print(f"  eventos en el registro: {len(registro)}")
    for linea, _ in registro:
        print(f"    {linea}")


if __name__ == "__main__":
    main()
