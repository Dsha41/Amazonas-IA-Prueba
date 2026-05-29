"""
analytics/attendance_tracker.py - Deteccion de atencion vendedor-cliente.

Un cliente se considera "atendido" cuando un vendedor se acerca a menos de
ATTENDANCE_DISTANCE_PX pixeles durante al menos ATTENDANCE_MIN_TIME_SEC segundos.
"""

import time
import math
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from .config import AnalyticsConfig

logger = logging.getLogger(__name__)


class AttendanceTracker:
    """Trackea interacciones vendedor-cliente basadas en proximidad temporal."""

    def __init__(self, config: AnalyticsConfig = None):
        cfg = config or AnalyticsConfig
        self._distance_px = cfg.ATTENDANCE_DISTANCE_PX
        self._min_time_sec = cfg.ATTENDANCE_MIN_TIME_SEC

        # IDs de vendedores activos (se pueden asignar manualmente o por ROI)
        self._seller_ids: Set[int] = set(cfg.SELLER_IDS)

        # Proximidades activas: (seller_id, client_id) -> timestamp_inicio
        self._active_proximities: Dict[Tuple[int, int], float] = {}

        # Atenciones confirmadas: client_id -> {seller_id, start_time, end_time, duration}
        self._attended: Dict[int, Dict] = {}

        # Historial de todas las interacciones
        self._interactions: List[Dict] = []

        # Total de personas atendidas (unicas)
        self._attended_count: int = 0

    def set_seller_ids(self, seller_ids: list):
        """Asigna manualmente los IDs de vendedores."""
        self._seller_ids = set(seller_ids)

    def add_seller(self, track_id: int):
        """Marca un track como vendedor."""
        self._seller_ids.add(track_id)

    def remove_seller(self, track_id: int):
        """Desmarca un track como vendedor."""
        self._seller_ids.discard(track_id)

    def is_seller(self, track_id: int) -> bool:
        """Retorna True si el track es un vendedor."""
        return track_id in self._seller_ids

    def update(self, active_tracks: Dict[int, Dict]) -> List[Dict]:
        """Actualiza el estado de proximidades con los tracks actuales.

        Args:
            active_tracks: dict de track_id -> track_data con al menos 'center'

        Returns:
            Lista de nuevas atenciones confirmadas en este frame
        """
        now = time.time()
        new_attendances = []

        # Separar vendedores y clientes
        sellers = {}
        clients = {}
        for tid, track in active_tracks.items():
            if tid in self._seller_ids:
                sellers[tid] = track
            else:
                clients[tid] = track

        # Evaluar proximidad entre cada vendedor y cada cliente
        current_close_pairs = set()
        for sid, seller in sellers.items():
            s_center = seller.get('center', (0, 0))
            for cid, client in clients.items():
                c_center = client.get('center', (0, 0))
                dist = math.hypot(s_center[0] - c_center[0], s_center[1] - c_center[1])

                if dist <= self._distance_px:
                    pair = (sid, cid)
                    current_close_pairs.add(pair)

                    if pair not in self._active_proximities:
                        # Nueva proximidad
                        self._active_proximities[pair] = now
                    else:
                        # Verificar si se cumple el tiempo minimo
                        start = self._active_proximities[pair]
                        elapsed = now - start
                        if elapsed >= self._min_time_sec and cid not in self._attended:
                            # Atencion confirmada
                            attendance = {
                                "seller_id": sid,
                                "client_id": cid,
                                "start_time": start,
                                "confirm_time": now,
                                "duration": elapsed,
                            }
                            self._attended[cid] = attendance
                            self._interactions.append(attendance)
                            self._attended_count += 1
                            new_attendances.append(attendance)
                            logger.info(
                                f"Atencion confirmada: vendedor {sid} -> cliente {cid} "
                                f"({elapsed:.1f}s)"
                            )

        # Limpiar proximidades rotas (se alejaron)
        stale_pairs = [p for p in self._active_proximities if p not in current_close_pairs]
        for pair in stale_pairs:
            del self._active_proximities[pair]

        return new_attendances

    def end_interaction(self, seller_id: int, client_id: int):
        """Registra el fin de una interaccion (cuando el cliente se va)."""
        pair = (seller_id, client_id)
        if pair in self._active_proximities:
            start = self._active_proximities.pop(pair)
            duration = time.time() - start
            # Actualizar la interaccion en el historial
            if client_id in self._attended:
                self._attended[client_id]["end_time"] = time.time()
                self._attended[client_id]["duration"] = duration

    def get_seller_interactions(self, seller_id: int) -> List[Dict]:
        """Retorna todas las interacciones de un vendedor."""
        return [i for i in self._interactions if i["seller_id"] == seller_id]

    def get_stats(self) -> Dict:
        """Retorna estadisticas para overlay/metadata."""
        return {
            "attended_count": self._attended_count,
            "active_proximities": len(self._active_proximities),
            "seller_count": len(self._seller_ids),
            "seller_ids": list(self._seller_ids),
        }

    def get_active_proximity_pairs(self) -> List[Tuple[int, int]]:
        """Retorna pares (seller_id, client_id) actualmente cercanos."""
        return list(self._active_proximities.keys())

    def is_attended(self, client_id: int) -> bool:
        """Retorna True si el cliente ya fue atendido."""
        return client_id in self._attended

    def remove_track(self, track_id: int):
        """Limpia datos de un track eliminado."""
        # Limpiar proximidades
        pairs_to_remove = [p for p in self._active_proximities
                          if track_id in p]
        for pair in pairs_to_remove:
            del self._active_proximities[pair]

    def reset(self):
        """Reinicia todo."""
        self._active_proximities.clear()
        self._attended.clear()
        self._interactions.clear()
        self._attended_count = 0
