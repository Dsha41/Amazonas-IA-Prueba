"""
utils/logger.py - Logging unificado a archivo JSONL para analitica retail.

Cada entrada:
    {"timestamp": "...", "event": "...", "data": {...}}

Eventos soportados:
    person_detected, person_attended, efficiency_award, stock_alert,
    demographic_classified
"""

import csv
import json
import os
import time
import datetime
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AnalyticsLogger:
    """Logger unificado que escribe eventos en formato JSON Lines."""

    def __init__(self, log_path: str = "output/analytics_log.jsonl",
                 csv_path: str = "output/visitas.csv"):
        self._log_path = log_path
        self._lock = threading.Lock()
        self._buffer: list = []
        self._flush_every = 10  # flush cada 10 eventos
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)

        # CSV de visitas: una fila por persona que entro y salio del area
        # con genero, edad y tiempo total de permanencia.
        self._csv_path = csv_path
        self._csv_lock = threading.Lock()
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        # Crear encabezado si el archivo no existe
        if not os.path.exists(self._csv_path):
            try:
                with open(self._csv_path, 'w', newline='',
                          encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow([
                        "Genero",
                        "Edad",
                        "Tiempo total de permanencia",
                    ])
            except Exception as e:
                logger.error(f"Error creando CSV de visitas: {e}")

    def log(self, event: str, data: Optional[Dict[str, Any]] = None):
        """Registra un evento en el log.

        Args:
            event: Tipo de evento (e.g., "person_detected")
            data: Datos adicionales del evento
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event,
            "data": data or {},
        }

        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._flush_every:
                self._flush()

    def log_person_detected(self, track_id: int, gender: str = "",
                            age_range: str = "", camera_id: Any = None):
        self.log("person_detected", {
            "track_id": track_id,
            "gender": gender,
            "age_range": age_range,
            "camera_id": str(camera_id) if camera_id else "",
        })

    def log_person_attended(self, seller_id: int, client_id: int,
                            duration: float, camera_id: Any = None):
        self.log("person_attended", {
            "seller_id": seller_id,
            "client_id": client_id,
            "duration_sec": round(duration, 1),
            "camera_id": str(camera_id) if camera_id else "",
        })

    def log_efficiency_award(self, seller_id: int, count: int,
                             avg_time: float, total_people: int):
        self.log("efficiency_award", {
            "seller_id": seller_id,
            "count": count,
            "avg_time_sec": round(avg_time, 1),
            "total_people": total_people,
        })

    def log_stock_alert(self, roi_name: str, status: str, fill_ratio: float):
        self.log("stock_alert", {
            "roi_name": roi_name,
            "status": status,
            "fill_ratio": round(fill_ratio, 2),
        })

    def log_demographic(self, track_id: int, gender: str, age_range: str):
        self.log("demographic_classified", {
            "track_id": track_id,
            "gender": gender,
            "age_range": age_range,
        })

    def log_visit_csv(self, gender: str, age_range: str,
                      total_seconds: float):
        """Registra una visita completa en el CSV.

        Columnas: Genero, Edad, Tiempo total de permanencia (HH:MM:SS).
        Se llama al detectar la salida del area de una persona.

        Se OMITE la fila si genero o edad son 'Desconocido': solo
        interesan visitas con clasificacion confiable.
        """
        g = (gender or "").strip()
        a = (age_range or "").strip()
        if not g or not a or g == "Desconocido" or a == "Desconocido":
            return

        if total_seconds < 0:
            total_seconds = 0
        secs = int(total_seconds)
        hh = secs // 3600
        mm = (secs % 3600) // 60
        ss = secs % 60
        duration_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

        try:
            with self._csv_lock:
                with open(self._csv_path, 'a', newline='',
                          encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow([g, a, duration_str])
        except Exception as e:
            logger.error(f"Error escribiendo CSV visita: {e}")

    def _flush(self):
        """Escribe el buffer al archivo."""
        if not self._buffer:
            return
        try:
            with open(self._log_path, 'a', encoding='utf-8') as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._buffer.clear()
        except Exception as e:
            logger.error(f"Error escribiendo analytics log: {e}")

    def flush(self):
        """Flush manual del buffer."""
        with self._lock:
            self._flush()

    def close(self):
        """Cierra el logger flusheando el buffer."""
        self.flush()
