"""
analytics/stock_monitor.py - Monitoreo de productos en estantes por ROI.

Define ROIs de producto y analiza si el estante esta lleno, bajo o agotado
comparando el histograma actual contra una imagen de referencia.
"""

import cv2
import numpy as np
import os
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple

from .config import AnalyticsConfig

logger = logging.getLogger(__name__)

# Estados posibles
STATUS_OK = "OK"           # > 70% lleno
STATUS_LOW = "BAJO"        # 30-70%
STATUS_EMPTY = "AGOTADO"   # < 30%


class ProductROI:
    """Representa un ROI de producto en el frame."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int, name: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.name = name
        self.status = STATUS_OK
        self.fill_ratio = 1.0
        self.reference_hist: Optional[np.ndarray] = None
        self.last_update: float = 0.0

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class StockMonitor:
    """Monitorea el nivel de stock en ROIs de producto.

    Compara histogramas del ROI actual vs referencia (estante lleno).
    Ejecuta el analisis en un thread separado para no bloquear el loop principal.
    """

    def __init__(self, config: AnalyticsConfig = None):
        cfg = config or AnalyticsConfig
        self._threshold_low = cfg.STOCK_THRESHOLD_LOW
        self._threshold_ok = cfg.STOCK_THRESHOLD_OK
        self._reference_dir = cfg.STOCK_REFERENCE_DIR
        self._alerts_log = cfg.STOCK_ALERTS_LOG

        # ROIs de producto
        self._rois: List[ProductROI] = []

        # Lock para acceso thread-safe
        self._lock = threading.Lock()

        # Historial de alertas
        self._alerts: List[Dict] = []

        # Control de frecuencia de analisis (cada N frames)
        self._analyze_every_n = 30  # cada 30 frames (~1 segundo a 30fps)
        self._frame_count = 0

        # Cargar ROIs desde config
        for roi_data in cfg.PRODUCT_ROIS:
            if len(roi_data) >= 5:
                self.add_roi(*roi_data[:5])

        os.makedirs(self._reference_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._alerts_log) or '.', exist_ok=True)

    def add_roi(self, x1: int, y1: int, x2: int, y2: int, name: str):
        """Agrega un ROI de producto."""
        roi = ProductROI(x1, y1, x2, y2, name)
        # Intentar cargar referencia
        ref_path = os.path.join(self._reference_dir, f"{name}_ref.jpg")
        if os.path.exists(ref_path):
            ref_img = cv2.imread(ref_path)
            if ref_img is not None:
                roi.reference_hist = self._compute_histogram(
                    ref_img[y1:y2, x1:x2] if ref_img.shape[0] > y2 and ref_img.shape[1] > x2
                    else ref_img
                )
                logger.info(f"Referencia cargada para ROI '{name}'")
        self._rois.append(roi)

    def set_reference(self, frame: np.ndarray, roi_index: int):
        """Captura una imagen de referencia (estante lleno) para un ROI."""
        if roi_index >= len(self._rois):
            return
        roi = self._rois[roi_index]
        crop = frame[roi.y1:roi.y2, roi.x1:roi.x2]
        if crop.size == 0:
            return
        roi.reference_hist = self._compute_histogram(crop)
        # Guardar referencia en disco
        ref_path = os.path.join(self._reference_dir, f"{roi.name}_ref.jpg")
        cv2.imwrite(ref_path, crop)
        logger.info(f"Referencia guardada para ROI '{roi.name}'")

    def update(self, frame: np.ndarray) -> List[Dict]:
        """Analiza los ROIs de producto (se ejecuta en el thread principal
        pero solo cada N frames para no impactar rendimiento).

        Args:
            frame: Frame completo BGR

        Returns:
            Lista de alertas nuevas generadas
        """
        self._frame_count += 1
        if self._frame_count % self._analyze_every_n != 0:
            return []

        if not self._rois:
            return []

        new_alerts = []
        h_frame, w_frame = frame.shape[:2]

        with self._lock:
            for roi in self._rois:
                # Validar que el ROI este dentro del frame
                if roi.x2 > w_frame or roi.y2 > h_frame:
                    continue

                crop = frame[roi.y1:roi.y2, roi.x1:roi.x2]
                if crop.size == 0:
                    continue

                current_hist = self._compute_histogram(crop)
                prev_status = roi.status

                if roi.reference_hist is not None:
                    # Comparar con referencia
                    correlation = cv2.compareHist(
                        roi.reference_hist, current_hist, cv2.HISTCMP_CORREL
                    )
                    # correlation: 1.0 = identico, 0.0 = totalmente diferente
                    roi.fill_ratio = max(0.0, min(1.0, correlation))
                else:
                    # Sin referencia: estimar por contenido visual (varianza de bordes)
                    roi.fill_ratio = self._estimate_fill_by_edges(crop)

                # Determinar estado
                if roi.fill_ratio >= self._threshold_ok:
                    roi.status = STATUS_OK
                elif roi.fill_ratio >= self._threshold_low:
                    roi.status = STATUS_LOW
                else:
                    roi.status = STATUS_EMPTY

                roi.last_update = time.time()

                # Generar alerta si cambio a AGOTADO
                if roi.status == STATUS_EMPTY and prev_status != STATUS_EMPTY:
                    alert = {
                        "roi_name": roi.name,
                        "status": roi.status,
                        "fill_ratio": roi.fill_ratio,
                        "timestamp": time.time(),
                    }
                    new_alerts.append(alert)
                    self._alerts.append(alert)
                    self._log_alert(alert)
                    logger.warning(f"STOCK AGOTADO: {roi.name} ({roi.fill_ratio:.0%})")

        return new_alerts

    def _compute_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Calcula histograma HSV normalizado del crop."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _estimate_fill_by_edges(self, crop: np.ndarray) -> float:
        """Estima nivel de llenado por densidad de bordes (sin referencia).
        Un estante lleno tiene mas bordes (productos) que uno vacio.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size if edges.size > 0 else 0
        # Normalizar: un estante lleno tipicamente tiene ~15-25% de bordes
        fill = min(1.0, edge_ratio / 0.20)
        return fill

    def _log_alert(self, alert: Dict):
        """Escribe alerta de stock en el log."""
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = (
                f"[{ts}] STOCK {alert['status']}: {alert['roi_name']} "
                f"({alert['fill_ratio']:.0%})\n"
            )
            with open(self._alerts_log, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception as e:
            logger.error(f"Error escribiendo log de stock: {e}")

    def get_rois(self) -> List[ProductROI]:
        """Retorna los ROIs con su estado actual."""
        with self._lock:
            return list(self._rois)

    def get_stats(self) -> Dict:
        """Retorna estadisticas para metadata."""
        with self._lock:
            rois_data = []
            for roi in self._rois:
                rois_data.append({
                    "name": roi.name,
                    "status": roi.status,
                    "fill_ratio": round(roi.fill_ratio, 2),
                    "rect": roi.rect,
                })
            return {
                "product_rois": rois_data,
                "total_alerts": len(self._alerts),
            }

    def reset(self):
        """Reinicia alertas (mantiene ROIs y referencias)."""
        with self._lock:
            self._alerts.clear()
            for roi in self._rois:
                roi.status = STATUS_OK
                roi.fill_ratio = 1.0
