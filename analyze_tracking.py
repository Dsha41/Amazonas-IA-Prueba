"""
analyze_tracking.py - Dos preguntas sobre la fiabilidad de los eventos.

  A. ¿En que MOMENTO dispara cada combinacion de umbrales? El conteo de
     eventos no dice si son los correctos: un evento en el instante
     equivocado cuenta igual que uno acertado.

  B. ¿Cuanto riesgo de DUPLICADO hay? ZoneEventTracker bloquea el segundo
     evento por client_id (self._confirmed), asi que un empleado que se
     aleja y vuelve NO duplica. El duplicado aparece si al CLIENTE le
     cambia el track_id a mitad de camino: para el sistema es otra
     persona y puede volver a disparar.

Trabaja sobre output/hummus_tracks.json, sin volver a correr YOLO.

Uso:
    python analyze_tracking.py
"""

import json
import math

import cv2
import numpy as np

from src.analityc.core.analytics.zone_event_tracker import ZoneEventTracker
from test_hummus_zones import load_zones, TRACKS_DUMP

# Un track nuevo que aparece a menos de esta distancia y dentro de esta
# ventana de frames desde que otro desaparecio, se considera la MISMA
# persona con identidad nueva (fallo de re-identificacion).
RELEVO_DIST_PX = 90
RELEVO_FRAMES = 8


def cargar():
    with open(TRACKS_DUMP, encoding="utf-8") as f:
        data = json.load(f)
    frames = []
    for fr in data["frames"]:
        frames.append({
            "frame": fr["frame"],
            "t": fr["t"],
            "tracks": {int(tid): {"center": tuple(t["center"]), "box": t["box"]}
                       for tid, t in fr["tracks"].items()},
        })
    return data.get("fps", 11.919), frames


def dentro(punto, poligono) -> bool:
    return cv2.pointPolygonTest(poligono, (int(punto[0]), int(punto[1])), False) >= 0


# ---------------------------------------------------------------------------
# A. Momento en que dispara cada combinacion
# ---------------------------------------------------------------------------

def momentos(nombre, zona, staff_poly, frames, referencia):
    print(f"\n{'=' * 66}")
    print(f"A. CUANDO DISPARA  --  {nombre}")
    if referencia:
        print(f"   momento real conocido: t={referencia:.1f}s")
    print(f"{'=' * 66}")
    print(f"  {'dist':>5} {'t_min':>6}  {'n':>3}  momentos de los eventos")

    for dist in (250, 300, 400):
        for mt in (1.0, 1.5, 2.0):
            tr = ZoneEventTracker(
                event_name=nombre, zone_polygon=zona,
                staff_side_polygon=staff_poly,
                distance_px=dist, min_time_sec=mt,
                require_staff_in_zone=False,
            )
            evs = []
            for fr in frames:
                evs.extend(tr.update(fr["tracks"], now=fr["t"]))

            if not evs:
                detalle = "(ninguno)"
            else:
                partes = []
                for e in evs:
                    t = e["confirm_time"]
                    marca = ""
                    if referencia is not None:
                        err = abs(t - referencia)
                        marca = "  <-- ACIERTA" if err <= 1.5 else f"  (falla por {err:.1f}s)"
                    partes.append(f"t={t:.1f}s cli={e['client_id']}{marca}")
                detalle = " | ".join(partes)
            print(f"  {dist:>5} {mt:>5.1f}s  {len(evs):>3}  {detalle}")


# ---------------------------------------------------------------------------
# B. Riesgo de duplicado por cambio de identidad
# ---------------------------------------------------------------------------

def duplicados(frames, zonas, staff_poly):
    print(f"\n{'=' * 66}")
    print("B. RIESGO DE DUPLICADO")
    print(f"{'=' * 66}")

    # Vida de cada track: primer y ultimo frame en que aparece
    vida = {}
    for i, fr in enumerate(frames):
        for tid, t in fr["tracks"].items():
            if tid not in vida:
                vida[tid] = {"desde": i, "hasta": i,
                             "pos_ini": t["center"], "pos_fin": t["center"]}
            else:
                vida[tid]["hasta"] = i
                vida[tid]["pos_fin"] = t["center"]

    duraciones = sorted((v["hasta"] - v["desde"] + 1) for v in vida.values())
    print(f"\n  tracks distintos en total .......... {len(vida)}")
    print(f"  duracion mediana de un track ....... {duraciones[len(duraciones) // 2]} frames")
    print(f"  tracks que duran menos de 10 frames  {sum(1 for d in duraciones if d < 10)}")

    # Relevos: un track muere y otro nace cerca, poco despues
    relevos = []
    for nuevo, vn in vida.items():
        if vn["desde"] == 0:
            continue
        for viejo, vv in vida.items():
            if viejo == nuevo or vv["hasta"] >= vn["desde"]:
                continue
            hueco = vn["desde"] - vv["hasta"]
            if hueco > RELEVO_FRAMES:
                continue
            d = math.hypot(vv["pos_fin"][0] - vn["pos_ini"][0],
                           vv["pos_fin"][1] - vn["pos_ini"][1])
            if d <= RELEVO_DIST_PX:
                relevos.append((viejo, nuevo, hueco, d,
                                frames[vn["desde"]]["t"]))

    print(f"\n  RELEVOS DE IDENTIDAD detectados: {len(relevos)}")
    print(f"  (un track desaparece y otro nace a menos de {RELEVO_DIST_PX}px")
    print(f"   dentro de {RELEVO_FRAMES} frames -> probablemente la misma persona)")
    for viejo, nuevo, hueco, d, t in sorted(relevos, key=lambda r: r[4])[:12]:
        print(f"    t={t:5.1f}s   {viejo} -> {nuevo}   "
              f"(hueco {hueco} frames, {d:.0f}px)")

    # Cuantos de esos relevos ocurren con el track DENTRO de una zona:
    # esos son los que pueden duplicar un evento.
    for nombre, zona in zonas:
        zpoly = np.array(zona, np.int32).reshape((-1, 1, 2))
        en_zona = [r for r in relevos
                   if dentro(vida[r[1]]["pos_ini"], zpoly)]
        print(f"\n  relevos dentro de '{nombre}': {len(en_zona)}")
        for viejo, nuevo, hueco, d, t in en_zona[:8]:
            print(f"    t={t:5.1f}s   {viejo} -> {nuevo}   ({d:.0f}px)")


def main():
    cfg = load_zones()
    fps, frames = cargar()
    print(f"Tracks: {len(frames)} frames a {fps:.1f} fps "
          f"({len(frames) / fps:.1f}s de video)")

    zonas = [
        ("toma_de_orden", cfg["zona_caja"], None),
        # frame 450 / 11.919 fps = 37.75s, la entrega documentada
        ("entrega_de_plato", cfg["zona_entrega"], 37.75),
    ]

    for nombre, zona, ref in zonas:
        momentos(nombre, zona, cfg["lado_staff"], frames, ref)

    duplicados(frames, [(n, z) for n, z, _ in zonas], cfg["lado_staff"])


if __name__ == "__main__":
    main()
