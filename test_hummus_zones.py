"""
test_hummus_zones.py - Prueba standalone de ZoneEventTracker sobre el video
de Hummus, SIN pasar por el servidor/websocket ni modificar el pipeline.

OBJETIVO DE ESTA VERSION: validar VISUALMENTE si la clasificacion
staff/cliente por lado del mostrador es correcta, antes de confiar en los
numeros de eventos. Los poligonos son una estimacion a ojo y son el
eslabon mas debil de todo el enfoque: si el rol se asigna mal, ajustar
umbrales no sirve de nada.

Genera frames anotados en output/hummus_debug/ mostrando:
  - los 3 poligonos (zona caja, zona entrega, lado staff)
  - cada track detectado con su id y el rol que se le asigno

Uso:
    python test_hummus_zones.py [--start N] [--frames N]
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from src.analityc.core.person_amazona_inference import PersonAmazonas
from src.analityc.core.analytics.zone_event_tracker import ZoneEventTracker

VIDEO_PATH = "toma de orden y entrega de plato.avi"
DEBUG_DIR = "output/hummus_debug"
ZONES_FILE = "hummus_zones.json"
# Volcado de los tracks detectados, frame a frame. Correr YOLO cuesta
# ~1s por frame en CPU; con este archivo se pueden probar umbrales y
# variantes de logica al instante, sin repetir la deteccion.
TRACKS_DUMP = "output/hummus_tracks.json"

# ---------------------------------------------------------------------------
# Coordenadas de zona sobre el frame de 960x576 (camara fija "AM-CAJA1").
#
# Estos valores son solo un PUNTO DE PARTIDA estimado a ojo. Si existe
# hummus_zones.json en la raiz, se usan los de ahi en su lugar -- ese
# archivo lo genera el editor visual de zonas.
#
# Viven aqui y NO en AnalyticsConfig porque son especificos de este video
# y AnalyticsConfig es config compartido de produccion. Si esto se valida
# y se lleva al servidor, ahi si tendria sentido moverlas.
#
# Toma de orden y entrega ocurren en el MISMO mostrador, en sub-areas
# distintas -- no son dos camaras ni dos salas.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "zona_caja": [(20, 260), (300, 220), (320, 420), (20, 480)],
    "zona_entrega": [(300, 150), (560, 110), (580, 260), (320, 300)],
    # Lado "cocina": un track es staff si su centro cae aqui, cliente si no.
    "lado_staff": [(300, 0), (960, 0), (960, 340), (560, 110), (300, 150)],
    # Sub-area del personal que SIRVE los platos. La cajera queda fuera:
    # esta en el centro del encuadre y emparejaba con cualquiera.
    "zona_servidor": None,
    "distance_px": 250.0,
    "min_time_sec": 1.5,
    # Tolerancia a oclusiones breves: sin ella, que alguien se cruce
    # por delante un instante reinicia el cronometro.
    "max_gap_sec": 0.5,
    # Deteccion de la orden por permanencia (ZoneDwellTracker).
    "orden_dwell_sec": 5.0,
    "orden_gap_sec": 1.0,
}


def load_zones():
    """Carga las zonas de hummus_zones.json si existe; si no, usa DEFAULTS."""
    if not os.path.exists(ZONES_FILE):
        print(f"({ZONES_FILE} no existe -- usando coordenadas por defecto)")
        return dict(DEFAULTS)

    with open(ZONES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    cfg = dict(DEFAULTS)
    for key in ("zona_caja", "zona_entrega", "lado_staff", "zona_servidor"):
        pts = data.get(key)
        if pts:
            if len(pts) < 3:
                raise SystemExit(
                    f"{ZONES_FILE}: '{key}' tiene {len(pts)} punto(s); "
                    "un poligono necesita al menos 3."
                )
            cfg[key] = [tuple(p) for p in pts]
    for key in ("distance_px", "min_time_sec", "max_gap_sec",
                "orden_dwell_sec", "orden_gap_sec"):
        if data.get(key) is not None:
            cfg[key] = float(data[key])

    print(f"Zonas cargadas de {ZONES_FILE}")
    return cfg


def draw_debug(frame, tracks, cfg, orden_tracker, entrega_tracker,
               frame_idx, video_time):
    """Dibuja poligonos y tracks con su rol asignado."""
    img = frame.copy()

    for poly, color, label in [
        (cfg["zona_caja"], (0, 255, 255), "ZONA CAJA"),
        (cfg["zona_entrega"], (0, 140, 255), "ZONA ENTREGA"),
        (cfg["lado_staff"], (255, 0, 255), "LADO STAFF"),
    ]:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, 2)
        cv2.putText(img, label, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    for tid, track in tracks.items():
        center = track.get('center', (0, 0))
        cx, cy = int(center[0]), int(center[1])

        # Mismo criterio que usan los ZoneEventTracker internamente.
        # Se consultan AMBAS zonas: antes solo se miraba la de caja, asi
        # que a quien estuviera en la zona de entrega lo etiquetaba
        # "fuera" y la imagen enganaba.
        if orden_tracker._is_staff(center):
            role, color = "STAFF", (255, 0, 255)
        elif orden_tracker._is_inside(center, orden_tracker._zone_polygon):
            role, color = "CLI-CAJA", (0, 255, 255)
        elif entrega_tracker._is_inside(center, entrega_tracker._zone_polygon):
            role, color = "CLI-ENTREGA", (0, 140, 255)
        else:
            role, color = "fuera", (150, 150, 150)

        box = track.get('box')
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (cx, cy), 5, color, -1)
        cv2.putText(img, f"{tid} {role}", (cx - 30, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    cv2.putText(img, f"frame {frame_idx}  t={video_time:.1f}s  tracks={len(tracks)}",
                (10, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=400,
                        help="Frame inicial (default 400; la entrega documentada esta ~450)")
    parser.add_argument("--frames", type=int, default=120,
                        help="Cuantos frames CONSECUTIVOS procesar (default 120 ~10s de video)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 11.9
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(DEBUG_DIR, exist_ok=True)

    cfg = load_zones()
    print(f"  proximidad: {cfg['distance_px']:.0f}px  |  "
          f"tiempo minimo: {cfg['min_time_sec']:.1f}s")

    print(f"Video: {total_frames} frames @ {fps:.1f} fps")
    print(f"Procesando frames {args.start}..{args.start + args.frames} CONSECUTIVOS.")
    print("  (consecutivos a proposito: BoTSORT asume frames seguidos; saltar")
    print("   frames reasigna track_ids y ningun par staff-cliente sobrevive)\n")

    processor = PersonAmazonas(device="cpu")

    # update_tracks() aplica un filtro de proximidad a self.roi_polygon
    # SIEMPRE, ignore o no el parametro activate_roi de process_frame()
    # (ver ~linea 1620). Por defecto es DEFAULT_ROI, una caja del layout
    # original de Amazonas que cubre x=500-1040. La zona de caja de este
    # video esta en x=20-320, o sea FUERA: sin este override los clientes
    # de la caja nunca serian tracks y toma_de_orden no podria dispararse.
    #
    # NO se arregla el pipeline aqui (archivo compartido/productivo):
    # se sobreescribe el ROI SOLO en esta instancia de prueba.
    processor.roi_polygon = np.array(
        [[0, 0], [960, 0], [960, 576], [0, 576]], np.int32
    )

    # Las dos variantes corren sobre EL MISMO pase de video, alimentadas
    # con los mismos tracks. Asi la unica diferencia entre A y B es la
    # logica, no el tracking (que varia entre corridas).
    #
    #   A -- la zona acota solo al cliente; el empleado puede estar en
    #        cualquier punto de lado_staff mientras quede a distance_px.
    #   B -- la zona es donde OCURRE la interaccion: empleado y cliente
    #        deben estar los dos dentro.
    def build(require_in_zone):
        return {
            "toma_de_orden": ZoneEventTracker(
                event_name="toma_de_orden", zone_polygon=cfg["zona_caja"],
                staff_side_polygon=cfg["lado_staff"],
                distance_px=cfg["distance_px"], min_time_sec=cfg["min_time_sec"],
                require_staff_in_zone=require_in_zone,
            ),
            "entrega_de_plato": ZoneEventTracker(
                event_name="entrega_de_plato", zone_polygon=cfg["zona_entrega"],
                staff_side_polygon=cfg["lado_staff"],
                distance_px=cfg["distance_px"], min_time_sec=cfg["min_time_sec"],
                require_staff_in_zone=require_in_zone,
            ),
        }

    modos = {
        "A (solo cliente en zona)": build(False),
        "B (ambos en zona)": build(True),
    }
    eventos = {m: {k: [] for k in trs} for m, trs in modos.items()}

    # Para dibujar y para el censo usamos los trackers del modo A, que
    # clasifican el rol sin exigir que el empleado este en la zona.
    ref_orden = modos["A (solo cliente en zona)"]["toma_de_orden"]
    ref_entrega = modos["A (solo cliente en zona)"]["entrega_de_plato"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    wall_start = time.time()
    processed = 0
    role_census = {"STAFF": 0, "CLI-CAJA": 0, "CLI-ENTREGA": 0, "fuera": 0}
    dump = []

    print(f"{'frame':>6} {'t_video':>8} {'tracks':>7} {'staff':>6} {'cli':>5} "
          f"{'A:ord':>6} {'A:ent':>6} {'B:ord':>6} {'B:ent':>6}")

    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx = args.start + i
        video_time = frame_idx / fps

        processor.process_frame(frame)

        # process_frame() hace _push_state()/_pop_state(): al retornar,
        # processor.active_tracks vuelve a ser el dict vacio del __init__.
        # Los tracks reales viven en el estado POR CAMARA (camera_id=1 es
        # el default de process_frame).
        tracks = processor.camera_states.get(1, {}).get('active_tracks', {})

        # Mapa track_id -> person_uuid del re-identificador facial.
        # Es atributo privado, pero es la unica via para comprobar si
        # ArcFace consigue unir los fragmentos que deja el tracking
        # cuando intercambia identidades. Se vuelca por frame porque el
        # mapa se poda: las entradas viejas se borran.
        reid = getattr(processor, "_reidentifier", None)
        uuids = dict(getattr(reid, "_track_to_person", {}) or {}) if reid else {}

        dump.append({
            "frame": frame_idx,
            "t": round(video_time, 3),
            "uuids": {str(k): v for k, v in uuids.items()},
            "tracks": {
                str(tid): {
                    "center": [float(t["center"][0]), float(t["center"][1])],
                    "box": [float(v) for v in t["box"]] if t.get("box") is not None else None,
                }
                for tid, t in tracks.items() if t.get("center") is not None
            },
        })

        hubo_evento = False
        for modo, trackers in modos.items():
            for nombre, tr in trackers.items():
                nuevos = tr.update(tracks, now=video_time)
                if nuevos:
                    eventos[modo][nombre].extend(nuevos)
                    hubo_evento = True

        n_staff = sum(1 for t in tracks.values()
                      if ref_orden._is_staff(t.get('center', (0, 0))))
        n_cli = len(tracks) - n_staff
        for t in tracks.values():
            c = t.get('center', (0, 0))
            if ref_orden._is_staff(c):
                role_census["STAFF"] += 1
            elif ref_orden._is_inside(c, ref_orden._zone_polygon):
                role_census["CLI-CAJA"] += 1
            elif ref_entrega._is_inside(c, ref_entrega._zone_polygon):
                role_census["CLI-ENTREGA"] += 1
            else:
                role_census["fuera"] += 1

        processed += 1
        if processed % 20 == 0 or hubo_evento:
            a = eventos["A (solo cliente en zona)"]
            b = eventos["B (ambos en zona)"]
            print(f"{frame_idx:>6} {video_time:>7.1f}s {len(tracks):>7} "
                  f"{n_staff:>6} {n_cli:>5} "
                  f"{len(a['toma_de_orden']):>6} {len(a['entrega_de_plato']):>6} "
                  f"{len(b['toma_de_orden']):>6} {len(b['entrega_de_plato']):>6}")

        # Guardar 1 de cada 10 frames anotados para revisar a ojo
        if processed % 10 == 0:
            out = draw_debug(frame, tracks, cfg, ref_orden,
                             ref_entrega, frame_idx, video_time)
            cv2.imwrite(f"{DEBUG_DIR}/f{frame_idx:04d}.jpg", out)

    cap.release()
    wall = time.time() - wall_start

    # El nombre lleva el rango para no pisar capturas anteriores: la
    # validacion fuera de muestra necesita conservar el tramo original.
    destino = TRACKS_DUMP.replace(
        ".json", f"_{args.start}_{args.start + args.frames}.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({"fps": fps, "frames": dump}, f)
    print(f"\nTracks volcados en {destino} ({len(dump)} frames)")

    print(f"Procesados {processed} frames en {wall:.1f}s "
          f"({wall / max(processed, 1):.2f}s/frame en CPU)")
    print("\nCenso de roles (suma sobre todos los frames):")
    for k, v in role_census.items():
        print(f"  {k:<12} {v}")
    print(f"\nFrames anotados en {DEBUG_DIR}/ -- revisar a ojo si los roles")
    print("estan bien asignados ANTES de creerle a los contadores de eventos.")

    for modo in modos:
        print(f"\n{'=' * 58}\nMODO {modo}\n{'=' * 58}")
        for nombre, evs in eventos[modo].items():
            print(f"\n  {nombre}: {len(evs)} evento(s)")
            for e in evs:
                print(f"    t={e['confirm_time']:.1f}s  staff {e['staff_id']} -> "
                      f"cliente {e['client_id']}  ({e['duration']:.1f}s)")


if __name__ == "__main__":
    main()
