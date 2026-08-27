"""
analytics/zone_event_tracker.py - Deteccion de eventos por zona + proximidad.

Generaliza el mecanismo de AttendanceTracker (proximidad + cronometro) para
casos donde:

  1. El rol (staff vs. cliente) se determina por el LADO del mostrador en el
     que cae el centro del track, no por una lista manual de IDs (SELLER_IDS
     no es practico en CCTV de mostrador: no hay forma de asignar IDs de
     antemano y el tracking los reasigna).
  2. El evento solo cuenta si el cliente esta dentro de una ZONA concreta
     (p.ej. "caja" o "entrega"), no en cualquier parte del frame.

Cada instancia representa UN tipo de evento en UNA zona. Para Hummus hacen
falta dos instancias independientes (orden y entrega) con estado propio,
para que el mismo cliente pueda disparar ambos eventos por separado -- a
diferencia de AttendanceTracker, que solo permite UN evento confirmado por
cliente en toda su vida (bloquea el segundo).

No liga el evento de orden con el de entrega del mismo cliente: el tracking
ya demostro romperse con facilidad en pruebas previas (ver CONTEXTO.md), asi
que ligar ambos eventos por track_id daria pares huerfanos frecuentes. Se
cuenta cada evento por separado.
"""

import time
from collections import deque
import math
import logging
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ZoneEventTracker:
    """Confirma un evento cuando un track 'staff' y un track 'cliente'
    permanecen a distancia de proximidad, con el cliente dentro de una zona,
    durante un tiempo minimo de video (no de reloj de pared).
    """

    def __init__(self,
                 event_name: str,
                 zone_polygon,
                 staff_side_polygon,
                 distance_px: float = 150.0,
                 min_time_sec: float = 1.5,
                 require_staff_in_zone: bool = False,
                 max_gap_sec: float = 0.5,
                 staff_pair_polygon=None):
        """
        staff_pair_polygon
            Sub-area de staff_side_polygon cuyos empleados pueden formar
            pareja para ESTE evento. Sirve cuando el personal no es
            intercambiable: en un mostrador de comida la cajera toma
            pedidos en la registradora y otra persona sirve los platos
            detras del mostrador, y son sitios distintos.

            Sin esto, la cajera -- que esta en el centro del encuadre y
            por tanto cerca de todo el mundo -- dispara entregas que
            nunca ocurrieron.

            Los empleados fuera de esta sub-area siguen siendo empleados
            (nunca se cuentan como clientes), simplemente no emparejan.
            Si es None, cualquier empleado puede emparejar.

        max_gap_sec
            Hueco maximo tolerado sin ver la pareja antes de reiniciar el
            cronometro. Sin tolerancia, que alguien se cruce por delante
            un instante basta para perder la cuenta y volver a empezar:
            medido en el video, una oclusion de 0.6s retraso una entrega
            casi 2 segundos respecto al momento real.

        require_staff_in_zone
            False (modo A) -- la zona acota SOLO al cliente. El empleado
                puede estar en cualquier punto de staff_side_polygon
                mientras quede a distance_px. Con un radio grande esto
                puede emparejar gente de tramos distintos del mostrador.
            True (modo B) -- la zona representa DONDE OCURRE la
                interaccion: empleado y cliente deben estar ambos dentro.
                Requiere que zone_polygon y staff_side_polygon se solapen,
                porque el empleado tiene que cumplir las dos condiciones.
        """
        self.event_name = event_name
        self._zone_polygon = np.array(zone_polygon, dtype=np.int32).reshape((-1, 1, 2))
        self._staff_polygon = np.array(staff_side_polygon, dtype=np.int32).reshape((-1, 1, 2))
        self._distance_px = distance_px
        self._min_time_sec = min_time_sec
        self._require_staff_in_zone = require_staff_in_zone
        self._max_gap_sec = max_gap_sec
        self._pair_polygon = (
            np.array(staff_pair_polygon, dtype=np.int32).reshape((-1, 1, 2))
            if staff_pair_polygon is not None else None
        )

        # (staff_id, client_id) -> {"inicio": t, "visto": t}
        self._active_proximities: Dict[Tuple[int, int], Dict[str, float]] = {}
        # client_id -> evento confirmado (una vez por cliente, para ESTE evento)
        self._confirmed: Dict[int, Dict] = {}
        self._history: List[Dict] = []

        # Diagnostico de cadencia. Si el intervalo entre frames procesados
        # supera a max_gap_sec, el cronometro se reinicia SIEMPRE y no se
        # confirma nada nunca -- en silencio. Pasa de verdad: en CPU el
        # servidor tarda ~1.5 s por frame y el valor calibrado sobre video
        # a 11.9 fps era 1.0 s. Se avisa una vez para que se vea.
        self._ultimo_now: Optional[float] = None
        self._aviso_cadencia_dado = False
        # Intervalos recientes entre frames, para ajustar la tolerancia
        # a la cadencia real. Ver _gap_efectivo().
        self._intervalos: Deque[float] = deque(maxlen=20)

    @staticmethod
    def _is_inside(point: Tuple[float, float], polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

    def _is_staff(self, center: Tuple[float, float]) -> bool:
        return self._is_inside(center, self._staff_polygon)

    def _gap_efectivo(self) -> float:
        """Tolerancia real, ajustada a la cadencia observada.

        max_gap_sec es un parametro TECNICO: cuanto tiempo puedo perder de
        vista a alguien antes de dar la interrupcion por definitiva. Su
        valor correcto depende de cada cuanto llegan los frames, y eso
        cambia radicalmente segun donde corra:

            analisis offline, tiempo de video a 11.9 fps -> 0.08 s/frame
            servidor en CPU, reloj de pared              -> ~1.5 s/frame

        Con 1.0 s configurado, lo primero funciona y lo segundo no detecta
        nada: cada frame se considera una interrupcion. Por eso la
        tolerancia efectiva nunca baja de 2.5 intervalos observados.

        A 11.9 fps eso da 0.21 s, por debajo del 1.0 s configurado, asi
        que el valor configurado manda y el comportamiento offline no
        cambia. En el servidor lento da ~3.7 s, que es lo que hace falta.
        """
        if len(self._intervalos) < 3:
            return self._max_gap_sec
        ordenados = sorted(self._intervalos)
        mediana = ordenados[len(ordenados) // 2]
        return max(self._max_gap_sec, 2.5 * mediana)

    def _revisar_cadencia(self, now: float):
        """Avisa si los frames llegan mas espaciados que la tolerancia.

        Cuando eso pasa, cada frame se considera una interrupcion, el
        cronometro vuelve a cero y NUNCA se confirma un evento. Sin este
        aviso el sintoma es "no detecta nada" sin ninguna pista del porque.
        """
        anterior = self._ultimo_now
        self._ultimo_now = now
        if anterior is None:
            return
        intervalo = now - anterior
        if intervalo > 0:
            self._intervalos.append(intervalo)
        # El aviso se da una sola vez, pero los intervalos se siguen
        # registrando siempre: la tolerancia adaptativa los necesita.
        if self._aviso_cadencia_dado:
            return
        if intervalo > self._gap_efectivo():
            self._aviso_cadencia_dado = True
            logger.warning(
                "[%s] los frames llegan cada %.1f s, mas que los %.1f s "
                "configurados en max_gap_sec. Sin compensar, el cronometro "
                "se reiniciaria en cada frame y no se confirmaria nada. Se "
                "esta usando una tolerancia efectiva ajustada a esa "
                "cadencia; sube max_gap_sec por encima de %.1f s para "
                "dejarlo explicito.",
                self.event_name, intervalo, self._max_gap_sec, intervalo,
            )

    def update(self, active_tracks: Dict[int, Dict], now: Optional[float] = None) -> List[Dict]:
        """Actualiza proximidades con los tracks del frame actual.

        Args:
            active_tracks: dict de track_id -> track_data con al menos 'center'.
            now: timestamp a usar para el cronometro. Por defecto time.time()
                (uso en vivo, igual que AttendanceTracker). Al reprocesar un
                video grabado offline, pasar el timestamp DEL VIDEO
                (frame_idx / fps) -- si no, el cronometro mide segundos de
                procesamiento en CPU, no segundos reales de la escena.

        Returns:
            Lista de nuevos eventos confirmados en este frame.
        """
        if now is None:
            now = time.time()
        self._revisar_cadencia(now)
        new_events = []

        staff = {}
        clients_in_zone = {}
        for tid, track in active_tracks.items():
            center = track.get('center', (0, 0))
            in_zone = self._is_inside(center, self._zone_polygon)

            if self._is_staff(center):
                # Es empleado: nunca cuenta como cliente, aunque no
                # pueda emparejar para este evento concreto.
                if (self._pair_polygon is not None
                        and not self._is_inside(center, self._pair_polygon)):
                    continue
                # Modo B: el empleado tambien tiene que estar en la zona.
                # Modo A: basta con que este del lado del personal.
                if in_zone or not self._require_staff_in_zone:
                    staff[tid] = track
            elif in_zone:
                clients_in_zone[tid] = track

        current_close_pairs = set()
        for sid, s_track in staff.items():
            s_center = s_track.get('center', (0, 0))
            for cid, c_track in clients_in_zone.items():
                c_center = c_track.get('center', (0, 0))
                dist = math.hypot(s_center[0] - c_center[0], s_center[1] - c_center[1])

                if dist <= self._distance_px:
                    pair = (sid, cid)
                    current_close_pairs.add(pair)

                    prox = self._active_proximities.get(pair)
                    if prox is None or (now - prox["visto"]) > self._gap_efectivo():
                        # Pareja nueva, o el hueco fue tan largo que la
                        # cercania anterior se da por terminada.
                        self._active_proximities[pair] = {"inicio": now, "visto": now}
                    else:
                        # Hueco corto (una oclusion): se mantiene el
                        # cronometro y se sigue acumulando.
                        prox["visto"] = now
                        start = prox["inicio"]
                        elapsed = now - start
                        if elapsed >= self._min_time_sec and cid not in self._confirmed:
                            event = {
                                "event": self.event_name,
                                "staff_id": sid,
                                "client_id": cid,
                                "start_time": start,
                                "confirm_time": now,
                                "duration": elapsed,
                            }
                            self._confirmed[cid] = event
                            self._history.append(event)
                            new_events.append(event)
                            logger.info(
                                f"[{self.event_name}] confirmado: staff {sid} -> "
                                f"cliente {cid} ({elapsed:.1f}s)"
                            )

        # Solo se descartan las parejas que llevan mas de max_gap_sec sin
        # verse. Borrarlas en cuanto fallan un frame es lo que hacia que
        # una oclusion breve reiniciara la cuenta.
        caducadas = [p for p, v in self._active_proximities.items()
                     if p not in current_close_pairs
                     and (now - v["visto"]) > self._gap_efectivo()]
        for pair in caducadas:
            del self._active_proximities[pair]

        return new_events

    def is_confirmed(self, client_id: int) -> bool:
        return client_id in self._confirmed

    def get_stats(self) -> Dict:
        return {
            "event_name": self.event_name,
            "confirmed_count": len(self._confirmed),
            "active_proximities": len(self._active_proximities),
        }

    def remove_track(self, track_id: int):
        pairs_to_remove = [p for p in self._active_proximities if track_id in p]
        for pair in pairs_to_remove:
            del self._active_proximities[pair]

    def reset(self):
        self._active_proximities.clear()
        self._confirmed.clear()
        self._history.clear()
