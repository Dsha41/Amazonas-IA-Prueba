"""
analyze_hummus.py - Diagnostico offline sobre los tracks ya detectados.

test_hummus_zones.py vuelca los tracks a output/hummus_tracks.json. Correr
YOLO cuesta ~1s por frame en CPU; este script reproduce esos tracks
guardados a traves del ZoneEventTracker REAL (no una copia de la logica,
para que no puedan divergir) y permite probar umbrales al instante.

Responde tres preguntas, en orden:

  1. ¿Coinciden alguna vez un empleado y un cliente dentro de la zona,
     en el mismo frame? Si nunca, ningun umbral puede ayudar.
  2. ¿A que distancia llegan a estar? Si el minimo es 300px, pedir 180
     no va a confirmar nada.
  3. ¿Cuanto dura la cercania? Si el par se rompe cada 3 frames, exigir
     1.5s (18 frames) es imposible aunque la distancia sea correcta.

Y al final barre combinaciones de distancia y tiempo para ver cuales
producirian eventos.

Uso:
    python analyze_hummus.py
"""

import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np

from src.analityc.core.analytics.zone_event_tracker import ZoneEventTracker
from test_hummus_zones import load_zones, TRACKS_DUMP


def is_inside(point, polygon) -> bool:
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0


def load_dump():
    if not os.path.exists(TRACKS_DUMP):
        raise SystemExit(
            f"No existe {TRACKS_DUMP}.\n"
            "Corre primero: python test_hummus_zones.py --start 300 --frames 400"
        )
    with open(TRACKS_DUMP, encoding="utf-8") as f:
        data = json.load(f)
    frames = []
    for fr in data["frames"]:
        tracks = {
            int(tid): {"center": (t["center"][0], t["center"][1]), "box": t["box"]}
            for tid, t in fr["tracks"].items()
        }
        frames.append({"frame": fr["frame"], "t": fr["t"], "tracks": tracks})
    return data.get("fps", 11.919), frames


def diagnose(nombre, zona, staff_poly, frames, fps):
    """Responde las tres preguntas para una zona."""
    zpoly = np.array(zona, np.int32).reshape((-1, 1, 2))
    spoly = np.array(staff_poly, np.int32).reshape((-1, 1, 2))

    frames_con_staff = 0
    frames_con_cliente = 0
    frames_con_ambos = 0
    min_dist_global = float("inf")
    dist_por_frame = []
    # (staff_id, client_id) -> racha actual / mejor racha, a varias distancias
    rachas = {d: defaultdict(int) for d in (150, 200, 250, 300, 400)}
    mejor_racha = {d: 0 for d in rachas}

    for fr in frames:
        # Criterio del modo A (el mas permisivo): al empleado no se le
        # exige estar en la zona. Sirve como cota superior -- si aqui no
        # coinciden nunca, en el modo B tampoco.
        staff, clientes = [], []
        for tid, t in fr["tracks"].items():
            c = t["center"]
            if is_inside(c, spoly):
                staff.append((tid, c))
            elif is_inside(c, zpoly):
                clientes.append((tid, c))

        if staff:
            frames_con_staff += 1
        if clientes:
            frames_con_cliente += 1
        if not (staff and clientes):
            for d in rachas:
                rachas[d].clear()
            continue
        frames_con_ambos += 1

        min_dist = float("inf")
        cercanos = {d: set() for d in rachas}
        for sid, sc in staff:
            for cid, cc in clientes:
                dist = math.hypot(sc[0] - cc[0], sc[1] - cc[1])
                min_dist = min(min_dist, dist)
                for d in rachas:
                    if dist <= d:
                        cercanos[d].add((sid, cid))

        dist_por_frame.append(min_dist)
        min_dist_global = min(min_dist_global, min_dist)

        for d in rachas:
            for par in list(rachas[d]):
                if par not in cercanos[d]:
                    del rachas[d][par]
            for par in cercanos[d]:
                rachas[d][par] += 1
                mejor_racha[d] = max(mejor_racha[d], rachas[d][par])

    total = len(frames)
    print(f"\n{'-' * 62}")
    print(f"ZONA: {nombre}   ({total} frames analizados)")
    print(f"{'-' * 62}")
    print(f"  frames con algun empleado visible ....... {frames_con_staff:>4} / {total}")
    print(f"  frames con algun cliente en la zona ..... {frames_con_cliente:>4} / {total}")
    print(f"  frames con LOS DOS a la vez ............. {frames_con_ambos:>4} / {total}")

    if frames_con_ambos == 0:
        print("\n  >> Nunca coinciden. Ningun umbral puede producir eventos aqui.")
        return

    dist_por_frame.sort()
    p50 = dist_por_frame[len(dist_por_frame) // 2]
    print(f"\n  distancia empleado-cliente mas corta .... {min_dist_global:>6.0f} px")
    print(f"  mediana de la distancia mas corta ....... {p50:>6.0f} px")

    print(f"\n  racha maxima de cercania continua:")
    for d in sorted(rachas):
        frames_racha = mejor_racha[d]
        segs = frames_racha / fps
        print(f"    a {d:>3} px .... {frames_racha:>3} frames seguidos = {segs:.2f}s")


def barrido(nombre, zona, staff_poly, frames, fps):
    """Cuenta eventos para varias combinaciones de distancia y tiempo."""
    distancias = [150, 200, 250, 300, 400]
    tiempos = [0.3, 0.5, 1.0, 1.5, 2.0]

    for modo, req in (("A (solo cliente en zona)", False), ("B (ambos en zona)", True)):
        print(f"\n  MODO {modo}  --  eventos detectados")
        print("    " + "tiempo\\dist".ljust(13) + "".join(f"{d:>7}px" for d in distancias))
        for mt in tiempos:
            fila = f"    {mt:>5.1f}s".ljust(17)
            for d in distancias:
                tr = ZoneEventTracker(
                    event_name=nombre, zone_polygon=zona,
                    staff_side_polygon=staff_poly,
                    distance_px=d, min_time_sec=mt,
                    require_staff_in_zone=req,
                )
                n = 0
                for fr in frames:
                    n += len(tr.update(fr["tracks"], now=fr["t"]))
                fila += f"{n:>9}"
            print(fila)


def main():
    cfg = load_zones()
    fps, frames = load_dump()
    print(f"Tracks: {len(frames)} frames a {fps:.1f} fps "
          f"({len(frames) / fps:.1f}s de video)")

    zonas = [
        ("toma_de_orden", cfg["zona_caja"]),
        ("entrega_de_plato", cfg["zona_entrega"]),
    ]

    for nombre, zona in zonas:
        diagnose(nombre, zona, cfg["lado_staff"], frames, fps)

    print(f"\n{'=' * 62}")
    print("BARRIDO DE UMBRALES")
    print(f"{'=' * 62}")
    for nombre, zona in zonas:
        print(f"\n### {nombre}")
        barrido(nombre, zona, cfg["lado_staff"], frames, fps)


if __name__ == "__main__":
    main()
