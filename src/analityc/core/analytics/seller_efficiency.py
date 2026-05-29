"""
analytics/seller_efficiency.py - Metricas de eficiencia de vendedores y premio horario.

Para cada vendedor registra:
  - Cantidad de personas atendidas (interacciones)
  - Tiempo promedio por interaccion
  - Ratio: personas_atendidas / personas_totales_en_zona

Cada EFFICIENCY_AWARD_INTERVAL_SEC genera un premio al mejor vendedor.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .config import AnalyticsConfig

logger = logging.getLogger(__name__)


class SellerMetrics:
    """Metricas acumuladas de un vendedor individual."""

    def __init__(self, seller_id: int):
        self.seller_id = seller_id
        self.interactions: List[Dict] = []
        self.total_attendance_time: float = 0.0
        self._current_interaction_start: Optional[float] = None

    @property
    def count(self) -> int:
        return len(self.interactions)

    @property
    def avg_time(self) -> float:
        if not self.interactions:
            return 0.0
        total = sum(i.get("duration", 0) for i in self.interactions)
        return total / len(self.interactions)

    def add_interaction(self, client_id: int, duration: float):
        self.interactions.append({
            "client_id": client_id,
            "duration": duration,
            "timestamp": time.time(),
        })
        self.total_attendance_time += duration

    def efficiency(self, total_people: int) -> float:
        if total_people <= 0:
            return 0.0
        return (self.count / total_people) * 100.0


class SellerEfficiency:
    """Gestor de metricas de eficiencia para todos los vendedores."""

    def __init__(self, config: AnalyticsConfig = None):
        cfg = config or AnalyticsConfig
        self._award_interval = cfg.EFFICIENCY_AWARD_INTERVAL_SEC
        self._award_banner_duration = cfg.AWARD_BANNER_DURATION_SEC
        self._awards_log_file = cfg.EFFICIENCY_AWARDS_LOG

        # Metricas por vendedor
        self._sellers: Dict[int, SellerMetrics] = {}

        # Control del premio horario
        self._last_award_time: float = time.time()
        self._current_award: Optional[Dict] = None
        self._award_display_until: float = 0.0

        # Historial de premios
        self._award_history: List[Dict] = []

        # Contadores por hora (se resetean cada premio)
        self._hourly_counts: Dict[int, int] = defaultdict(int)

    def register_seller(self, seller_id: int):
        """Registra un vendedor si no existe."""
        if seller_id not in self._sellers:
            self._sellers[seller_id] = SellerMetrics(seller_id)

    def record_interaction(self, seller_id: int, client_id: int, duration: float):
        """Registra una interaccion vendedor-cliente completada."""
        self.register_seller(seller_id)
        self._sellers[seller_id].add_interaction(client_id, duration)
        self._hourly_counts[seller_id] += 1

    def check_award(self, total_people: int) -> Optional[Dict]:
        """Verifica si es momento de otorgar un premio de eficiencia.

        Args:
            total_people: Total de personas unicas en la escena

        Returns:
            Dict con info del premio si se otorgo, None si no
        """
        now = time.time()

        # Verificar si ya paso el intervalo
        if now - self._last_award_time < self._award_interval:
            return None

        self._last_award_time = now

        # Si no hay vendedores con interacciones, no hay premio
        if not self._hourly_counts:
            return None

        # Encontrar al mejor vendedor de la hora
        best_seller_id = None
        best_count = 0
        best_avg_time = float('inf')

        for sid, count in self._hourly_counts.items():
            if count > best_count:
                best_seller_id = sid
                best_count = count
                best_avg_time = self._sellers[sid].avg_time
            elif count == best_count and sid in self._sellers:
                # Empate: premia al de menor tiempo promedio
                avg = self._sellers[sid].avg_time
                if avg < best_avg_time:
                    best_seller_id = sid
                    best_avg_time = avg

        if best_seller_id is None:
            return None

        award = {
            "seller_id": best_seller_id,
            "count": best_count,
            "avg_time": best_avg_time,
            "timestamp": now,
            "total_people": total_people,
        }

        self._current_award = award
        self._award_display_until = now + self._award_banner_duration
        self._award_history.append(award)

        # Log a archivo
        self._log_award(award)

        # Resetear contadores horarios
        self._hourly_counts.clear()

        logger.info(
            f"Premio eficiencia: vendedor #{best_seller_id} con {best_count} "
            f"atenciones (prom: {best_avg_time:.1f}s)"
        )
        return award

    def _log_award(self, award: Dict):
        """Escribe el premio en el archivo de log."""
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = (
                f"[{ts}] VENDEDOR DEL MOMENTO: #{award['seller_id']} | "
                f"Atenciones: {award['count']} | "
                f"Prom: {award['avg_time']:.1f}s | "
                f"Personas totales: {award['total_people']}\n"
            )
            import os
            os.makedirs(os.path.dirname(self._awards_log_file) or '.', exist_ok=True)
            with open(self._awards_log_file, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception as e:
            logger.error(f"Error escribiendo log de premio: {e}")

    def get_active_award(self) -> Optional[Dict]:
        """Retorna el premio activo si aun debe mostrarse en pantalla."""
        if self._current_award and time.time() <= self._award_display_until:
            return self._current_award
        return None

    def get_dashboard(self, total_people: int) -> List[Dict]:
        """Retorna datos del mini-dashboard de vendedores para overlay.

        Returns:
            Lista de dicts con info de cada vendedor, ordenada por atenciones desc
        """
        dashboard = []
        for sid, metrics in self._sellers.items():
            dashboard.append({
                "seller_id": sid,
                "count": metrics.count,
                "avg_time_sec": metrics.avg_time,
                "avg_time_min": metrics.avg_time / 60.0,
                "efficiency_pct": metrics.efficiency(total_people),
                "hourly_count": self._hourly_counts.get(sid, 0),
            })
        # Ordenar por cantidad de atenciones descendente
        dashboard.sort(key=lambda x: x["count"], reverse=True)
        return dashboard

    def get_stats(self) -> Dict:
        """Retorna estadisticas para metadata."""
        return {
            "sellers": {
                sid: {
                    "count": m.count,
                    "avg_time": round(m.avg_time, 1),
                    "hourly_count": self._hourly_counts.get(sid, 0),
                }
                for sid, m in self._sellers.items()
            },
            "awards_given": len(self._award_history),
            "time_to_next_award": max(
                0,
                self._award_interval - (time.time() - self._last_award_time)
            ),
        }

    def reset(self):
        """Reinicia todo."""
        self._sellers.clear()
        self._hourly_counts.clear()
        self._current_award = None
        self._award_history.clear()
        self._last_award_time = time.time()
