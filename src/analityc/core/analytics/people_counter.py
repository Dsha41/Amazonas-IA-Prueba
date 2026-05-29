"""
analytics/people_counter.py - Conteo unico de personas.

Usa track IDs del sistema de tracking (BoTSORT) para asegurar que
cada persona se cuente una sola vez, incluso si sale y vuelve a entrar.
"""

import time
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)


class PeopleCounter:
    """Contador acumulado de personas unicas que han aparecido en la escena."""

    def __init__(self):
        # Set de todos los track_ids que hemos visto alguna vez
        self._seen_ids: Set[int] = set()
        # Personas actualmente visibles en el frame
        self._active_count: int = 0
        # Timestamp de la primera y ultima deteccion
        self._first_seen_time: float = 0.0
        self._last_seen_time: float = 0.0

    @property
    def total_unique(self) -> int:
        """Total de personas unicas detectadas."""
        return len(self._seen_ids)

    @property
    def active_count(self) -> int:
        """Personas actualmente visibles."""
        return self._active_count

    def register_track(self, track_id: int) -> bool:
        """Registra un track_id. Retorna True si es una persona nueva.

        Args:
            track_id: ID unico asignado por el tracker

        Returns:
            True si es la primera vez que vemos este track_id
        """
        now = time.time()
        if self._first_seen_time == 0.0:
            self._first_seen_time = now
        self._last_seen_time = now

        is_new = track_id not in self._seen_ids
        if is_new:
            self._seen_ids.add(track_id)
        return is_new

    def update_active(self, active_track_ids: set):
        """Actualiza el conteo de personas activas en el frame actual.

        Args:
            active_track_ids: Set de track_ids visibles en este frame
        """
        self._active_count = len(active_track_ids)

    def get_stats(self) -> Dict:
        """Retorna estadisticas para overlay/metadata."""
        return {
            "total_unique": self.total_unique,
            "active_now": self._active_count,
            "first_seen": self._first_seen_time,
            "last_seen": self._last_seen_time,
        }

    def reset(self):
        """Reinicia contadores."""
        self._seen_ids.clear()
        self._active_count = 0
        self._first_seen_time = 0.0
        self._last_seen_time = 0.0
