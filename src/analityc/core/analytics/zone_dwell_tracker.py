"""
analytics/zone_dwell_tracker.py - Deteccion de eventos por PERMANENCIA.

Complementa a ZoneEventTracker, que detecta por proximidad. La eleccion
entre uno y otro no es de gusto: depende de la forma que tenga el evento
en los datos.

  Proximidad (ZoneEventTracker) sirve cuando hay un ACERCAMIENTO medible:
  el empleado se aproxima, ocurre algo, se separa. En el video de Hummus
  la entrega de plato dibuja una V limpia en la distancia, con minimo en
  el instante de la entrega.

  Permanencia (esta clase) sirve cuando los dos ya estan ahi y no se
  mueven. La toma de orden en la caja dura 20-30 segundos con el cliente
  y el cajero a distancia practicamente constante: no hay ningun instante
  que destaque, asi que ningun umbral de proximidad puede encontrarlo. Lo
  que si es evidente en los datos es que el cliente PERMANECE.

Tolera huecos cortos a proposito: un cliente parado justo en el borde del
poligono entra y sale del area por ruido de deteccion de unos pocos
pixeles. Sin tolerancia, una sola persona se contaria como varias visitas.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ZoneDwellTracker:
    """Confirma un evento cuando un cliente permanece en una zona el
    tiempo suficiente. No exige que haya un empleado cerca: en un
    mostrador el personal esta siempre presente, asi que su cercania no
    aporta informacion.
    """

    def __init__(self,
                 event_name: str,
                 zone_polygon,
                 staff_side_polygon,
                 min_dwell_sec: float = 5.0,
                 max_gap_sec: float = 1.0):
        """
        min_dwell_sec
            Segundos que el cliente debe permanecer para confirmar.
        max_gap_sec
            Hueco maximo tolerado sin verlo antes de dar la visita por
            terminada. Un hueco mayor reinicia el cronometro.
        """
        self.event_name = event_name
        self._zone_polygon = np.array(zone_polygon, dtype=np.int32).reshape((-1, 1, 2))
        self._staff_polygon = np.array(staff_side_polygon, dtype=np.int32).reshape((-1, 1, 2))
        self._min_dwell_sec = min_dwell_sec
        self._max_gap_sec = max_gap_sec

        # client_id -> {"inicio": t, "visto": t}
        self._visitas: Dict[int, Dict[str, float]] = {}
        self._confirmed: Dict[int, Dict] = {}
        self._history: List[Dict] = []

    @staticmethod
    def _is_inside(point: Tuple[float, float], polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

    def _is_staff(self, center: Tuple[float, float]) -> bool:
        return self._is_inside(center, self._staff_polygon)

    def update(self, active_tracks: Dict[int, Dict],
               now: Optional[float] = None) -> List[Dict]:
        """Actualiza las permanencias con los tracks del frame actual.

        Args:
            active_tracks: track_id -> datos con al menos 'center'.
            now: timestamp para el cronometro. Al reprocesar un video
                grabado hay que pasar el tiempo DEL VIDEO; con
                time.time() se medirian segundos de CPU.

        Returns:
            Lista de permanencias confirmadas en este frame.
        """
        if now is None:
            now = time.time()
        new_events = []

        presentes = set()
        for tid, track in active_tracks.items():
            center = track.get('center', (0, 0))
            if self._is_staff(center):
                continue
            if not self._is_inside(center, self._zone_polygon):
                continue
            presentes.add(tid)

            visita = self._visitas.get(tid)
            if visita is None or (now - visita["visto"]) > self._max_gap_sec:
                # Primera vez, o el hueco fue tan largo que la visita
                # anterior se da por terminada: arranca una nueva.
                self._visitas[tid] = {"inicio": now, "visto": now}
                continue

            visita["visto"] = now
            permanencia = now - visita["inicio"]
            if permanencia >= self._min_dwell_sec and tid not in self._confirmed:
                event = {
                    "event": self.event_name,
                    "client_id": tid,
                    "start_time": visita["inicio"],
                    "confirm_time": now,
                    "duration": permanencia,
                }
                self._confirmed[tid] = event
                self._history.append(event)
                new_events.append(event)
                logger.info(
                    f"[{self.event_name}] confirmado: cliente {tid} "
                    f"permanecio {permanencia:.1f}s"
                )

        # Las visitas que llevan mas de max_gap_sec sin verse se descartan.
        # No se borran en cuanto desaparecen, justamente para tolerar el
        # parpadeo de deteccion en el borde del poligono.
        caducadas = [tid for tid, v in self._visitas.items()
                     if tid not in presentes and (now - v["visto"]) > self._max_gap_sec]
        for tid in caducadas:
            del self._visitas[tid]

        return new_events

    def is_confirmed(self, client_id: int) -> bool:
        return client_id in self._confirmed

    def get_stats(self) -> Dict:
        return {
            "event_name": self.event_name,
            "confirmed_count": len(self._confirmed),
            "visitas_en_curso": len(self._visitas),
        }

    def remove_track(self, track_id: int):
        self._visitas.pop(track_id, None)

    def reset(self):
        self._visitas.clear()
        self._confirmed.clear()
        self._history.clear()
