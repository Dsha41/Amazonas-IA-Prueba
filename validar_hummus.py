"""
validar_hummus.py - Validacion fuera de muestra.

Los umbrales de abajo se fijaron mirando SOLO el tramo 300-700. Este
script los aplica tal cual a un tramo distinto, sin reajustar nada. Esa
es toda la diferencia entre "encontre unos umbrales que funcionan" y
"ajuste hasta que funcionara": si hay que tocarlos para que acierte en el
tramo nuevo, no generalizan, y hay que decirlo.

Uso:
    python validar_hummus.py output/hummus_tracks_700_1035.json
"""

import json
import sys

from src.analityc.core.analytics.zone_event_tracker import ZoneEventTracker
from src.analityc.core.analytics.zone_dwell_tracker import ZoneDwellTracker
from test_hummus_zones import load_zones

# ---------------------------------------------------------------------------
# PARAMETROS CONGELADOS. No tocar al ver los resultados: ese es el punto.
# ---------------------------------------------------------------------------
# El elegido se lee de hummus_zones.json. La alternativa se mantiene
# escrita aqui a proposito: sirve de contraste contra el elegido.
ALTERNATIVA = ("proximidad 300px / 2.0s", 300, 2.0)


def cargar(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    frames = [
        {
            "t": fr["t"],
            "frame": fr["frame"],
            "tracks": {int(tid): {"center": tuple(t["center"])}
                       for tid, t in fr["tracks"].items()},
        }
        for fr in data["frames"]
    ]
    return data.get("fps", 11.919), frames


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output/hummus_tracks_700_1035.json"
    cfg = load_zones()
    fps, frames = cargar(path)

    t0, t1 = frames[0]["t"], frames[-1]["t"]
    print(f"Tramo de validacion: {path}")
    print(f"  {len(frames)} frames  ->  t={t0:.1f}s a t={t1:.1f}s "
          f"({t1 - t0:.1f}s de video)\n")

    print("=" * 64)
    print("ENTREGA DE PLATO  (proximidad)")
    print("=" * 64)
    candidatos = [
        ("elegido {:.0f}px / {:.1f}s".format(
            cfg["distance_px"], cfg["min_time_sec"]),
         cfg["distance_px"], cfg["min_time_sec"]),
        ALTERNATIVA,
    ]
    for etiqueta, dist, mt in candidatos:
        tr = ZoneEventTracker(
            event_name="entrega_de_plato", zone_polygon=cfg["zona_entrega"],
            staff_side_polygon=cfg["lado_staff"],
            distance_px=dist, min_time_sec=mt, require_staff_in_zone=False,
            max_gap_sec=cfg["max_gap_sec"],
            staff_pair_polygon=cfg.get("zona_servidor"),
        )
        evs = []
        for fr in frames:
            evs.extend(tr.update(fr["tracks"], now=fr["t"]))
        print(f"\n  {etiqueta}: {len(evs)} evento(s)")
        for e in evs:
            print(f"    t={e['confirm_time']:.1f}s  staff {e['staff_id']} -> "
                  f"cliente {e['client_id']}  ({e['duration']:.1f}s cerca)")

    print("\n" + "=" * 64)
    print("TOMA DE ORDEN  (permanencia {:.0f}s)".format(cfg["orden_dwell_sec"]))
    print("=" * 64)
    tr = ZoneDwellTracker(
        event_name="toma_de_orden", zone_polygon=cfg["zona_caja"],
        staff_side_polygon=cfg["lado_staff"],
        min_dwell_sec=cfg["orden_dwell_sec"],
        max_gap_sec=cfg["orden_gap_sec"],
    )
    evs = []
    for fr in frames:
        evs.extend(tr.update(fr["tracks"], now=fr["t"]))
    print(f"\n  {len(evs)} evento(s)")
    for e in evs:
        print(f"    t={e['confirm_time']:.1f}s  cliente {e['client_id']}  "
              f"(llego en t={e['start_time']:.1f}s)")

    print("\n" + "-" * 64)
    print("Para juzgar esto hace falta mirar el video en esos instantes.")
    print("Un evento en un momento sin interaccion es un falso positivo;")
    print("una interaccion real sin evento es un falso negativo. El conteo")
    print("por si solo no dice cual es cual.")


if __name__ == "__main__":
    main()
