"""
utils/overlay.py - Funciones de dibujo en frame para analitica retail.

Paneles semi-transparentes con contadores demograficos, eficiencia
de vendedores, indicadores de stock y banner de premios.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple


# ── Colores ──
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)
COLOR_BLUE    = (255, 180, 0)
COLOR_PINK    = (180, 0, 255)
COLOR_CYAN    = (0, 255, 180)
COLOR_YELLOW  = (0, 255, 255)
COLOR_GREEN   = (0, 200, 0)
COLOR_RED     = (0, 0, 255)
COLOR_ORANGE  = (0, 165, 255)
COLOR_GOLD    = (0, 215, 255)
COLOR_GRAY    = (120, 120, 120)

# Semaforo de stock
STOCK_COLORS = {
    "OK":      COLOR_GREEN,
    "BAJO":    COLOR_ORANGE,
    "AGOTADO": COLOR_RED,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.45
FONT_MEDIUM = 0.55
FONT_LARGE = 0.7
FONT_XLARGE = 0.9


def _draw_panel(image: np.ndarray, x: int, y: int, w: int, h: int,
                alpha: float = 0.7, color: tuple = COLOR_BLACK) -> np.ndarray:
    """Dibuja un panel semi-transparente."""
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def draw_demographics_panel(image: np.ndarray, counts: Dict, x: int = 10, y: int = 10) -> np.ndarray:
    """Panel lateral izquierdo: contadores demograficos.

    Args:
        counts: {"gender": {"Hombre": N, "Mujer": N}, "age": {"0-12": N, ...}, "total_classified": N}
    """
    gender = counts.get("gender", {})
    age = counts.get("age", {})
    total = counts.get("total_classified", 0)

    lines = [
        f"Demograficas ({total})",
        f"  Hombres: {gender.get('Hombre', 0)}",
        f"  Mujeres: {gender.get('Mujer', 0)}",
        "",
        "Rango de Edad:",
    ]

    age_ranges = ["0-12", "13-17", "18-25", "26-35", "36-50", "51-65", "65+"]
    for ar in age_ranges:
        count = age.get(ar, 0)
        if count > 0:
            lines.append(f"  {ar}: {count}")

    line_h = 20
    panel_h = 12 + line_h * len(lines) + 12
    panel_w = 200

    image = _draw_panel(image, x, y, panel_w, panel_h, 0.75)

    ty = y + 12 + 14
    for i, line in enumerate(lines):
        if line == "":
            ty += 6
            continue
        color = COLOR_WHITE if i == 0 else (200, 200, 200)
        scale = FONT_MEDIUM if i == 0 else FONT_SMALL
        thickness = 2 if i == 0 else 1
        cv2.putText(image, line, (x + 8, ty), FONT, scale, color, thickness)
        ty += line_h

    return image


def draw_people_total(image: np.ndarray, total_unique: int, attended: int,
                      active_now: int, x: int = -1, y: int = 10) -> np.ndarray:
    """Parte superior central: personas totales + atendidas.

    Si x=-1, centra automaticamente.
    """
    text1 = f"Personas totales: {total_unique}"
    text2 = f"Atendidas: {attended}/{total_unique}"
    text3 = f"Activas: {active_now}"

    combined = f"{text1}  |  {text2}  |  {text3}"
    (tw, th), _ = cv2.getTextSize(combined, FONT, FONT_MEDIUM, 2)

    if x == -1:
        x = (image.shape[1] - tw - 24) // 2

    panel_w = tw + 24
    panel_h = th + 20

    image = _draw_panel(image, x, y, panel_w, panel_h, 0.75)
    cv2.putText(image, combined, (x + 12, y + th + 10), FONT, FONT_MEDIUM, COLOR_WHITE, 2)

    return image


def draw_products_total(image: np.ndarray, total_unique: int, active_now: int,
                        by_sku: Optional[Dict[str, int]] = None,
                        x: int = -1, y: int = 50) -> np.ndarray:
    """Panel superior: productos cosmeticos detectados.

    Args:
        total_unique: cantidad unica de productos vistos a lo largo del tiempo
                      (cuenta cada track_id una sola vez).
        active_now:   productos visibles en el frame actual.
        by_sku:       dict opcional {nombre_sku: cantidad_unica}. Se listan
                      los 3 SKUs mas frecuentes.
        x:            posicion horizontal. -1 = centrado.
        y:            posicion vertical (default debajo de Personas totales).
    """
    text1 = f"Productos totales: {total_unique}"
    text2 = f"Activos: {active_now}"
    combined = f"{text1}  |  {text2}"

    (tw, th), _ = cv2.getTextSize(combined, FONT, FONT_MEDIUM, 2)

    if x == -1:
        x = (image.shape[1] - tw - 24) // 2

    panel_w = tw + 24
    panel_h = th + 20

    image = _draw_panel(image, x, y, panel_w, panel_h, 0.75, color=(40, 40, 100))
    cv2.putText(image, combined, (x + 12, y + th + 10),
                FONT, FONT_MEDIUM, COLOR_ORANGE, 2)

    # Listar top-3 SKUs debajo del panel principal
    if by_sku:
        top = sorted(by_sku.items(), key=lambda kv: kv[1], reverse=True)[:3]
        line_h = 18
        extra_h = 8 + line_h * len(top) + 8
        image = _draw_panel(image, x, y + panel_h + 2, panel_w, extra_h, 0.7)
        ty = y + panel_h + 2 + 8 + 12
        for sku, cnt in top:
            label = sku if len(sku) <= 28 else sku[:25] + '...'
            cv2.putText(image, f"  {label}: {cnt}", (x + 12, ty),
                        FONT, FONT_SMALL, COLOR_WHITE, 1)
            ty += line_h

    return image


def draw_seller_dashboard(image: np.ndarray, dashboard: List[Dict],
                          x: int = -1, y: int = 10) -> np.ndarray:
    """Panel lateral derecho: eficiencia de vendedores.

    Si x=-1, alinea a la derecha.
    """
    if not dashboard:
        return image

    lines = ["Eficiencia Vendedores"]
    for entry in dashboard[:5]:  # Max 5 vendedores en el panel
        sid = entry["seller_id"]
        count = entry["count"]
        avg_min = entry["avg_time_min"]
        eff = entry["efficiency_pct"]
        lines.append(f"  #{sid}: {count} aten | Prom: {avg_min:.1f}min | Ef: {eff:.0f}%")

    line_h = 20
    panel_w = 340
    panel_h = 12 + line_h * len(lines) + 12

    if x == -1:
        x = image.shape[1] - panel_w - 10

    image = _draw_panel(image, x, y, panel_w, panel_h, 0.75)

    ty = y + 12 + 14
    for i, line in enumerate(lines):
        color = COLOR_GOLD if i == 0 else (200, 200, 200)
        scale = FONT_MEDIUM if i == 0 else FONT_SMALL
        thickness = 2 if i == 0 else 1
        cv2.putText(image, line, (x + 8, ty), FONT, scale, color, thickness)
        ty += line_h

    return image


def draw_award_banner(image: np.ndarray, award: Dict) -> np.ndarray:
    """Banner centrado grande con fondo dorado semi-transparente.

    Args:
        award: {"seller_id": int, "count": int, "avg_time": float}
    """
    h, w = image.shape[:2]

    banner_w = min(500, w - 40)
    banner_h = 100
    bx = (w - banner_w) // 2
    by = (h - banner_h) // 2

    # Fondo dorado semi-transparente
    image = _draw_panel(image, bx, by, banner_w, banner_h, 0.8, (0, 180, 220))

    # Borde dorado
    cv2.rectangle(image, (bx, by), (bx + banner_w, by + banner_h), COLOR_GOLD, 3)

    # Texto del premio
    title = "VENDEDOR DEL MOMENTO"
    sid = award.get("seller_id", 0)
    count = award.get("count", 0)

    cv2.putText(image, title, (bx + 20, by + 35), FONT, FONT_LARGE, COLOR_WHITE, 2)
    cv2.putText(
        image, f"Vendedor #{sid}", (bx + 20, by + 60),
        FONT, FONT_MEDIUM, COLOR_WHITE, 2
    )
    cv2.putText(
        image, f"Atenciones esta hora: {count}", (bx + 20, by + 85),
        FONT, FONT_SMALL, (220, 220, 220), 1
    )

    return image


def draw_stock_indicators(image: np.ndarray, rois: list) -> np.ndarray:
    """Dibuja indicadores de semaforo sobre cada ROI de producto.

    Args:
        rois: Lista de ProductROI con .rect, .status, .fill_ratio, .name
    """
    for roi in rois:
        x1, y1, x2, y2 = roi.rect
        color = STOCK_COLORS.get(roi.status, COLOR_GRAY)

        # Borde del ROI con color segun estado
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Circulo de estado (semaforo)
        cx = (x1 + x2) // 2
        cy = y1 - 15 if y1 > 25 else y2 + 15
        cv2.circle(image, (cx, cy), 10, color, -1)
        cv2.circle(image, (cx, cy), 10, COLOR_WHITE, 1)

        # Etiqueta
        label = f"{roi.name}: {roi.status} ({roi.fill_ratio:.0%})"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SMALL, 1)
        lx = max(0, cx - tw // 2)
        ly = cy - 18 if cy > 30 else cy + 25
        cv2.rectangle(image, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), COLOR_BLACK, -1)
        cv2.putText(image, label, (lx, ly), FONT, FONT_SMALL, color, 1)

    return image


def draw_proximity_lines(image: np.ndarray, pairs: list,
                         active_tracks: dict) -> np.ndarray:
    """Dibuja lineas de proximidad vendedor-cliente.

    Args:
        pairs: Lista de (seller_id, client_id)
        active_tracks: Dict de track_id -> track_data con 'center'
    """
    for sid, cid in pairs:
        s_track = active_tracks.get(sid)
        c_track = active_tracks.get(cid)
        if s_track and c_track:
            s_center = tuple(int(v) for v in s_track['center'])
            c_center = tuple(int(v) for v in c_track['center'])
            cv2.line(image, s_center, c_center, COLOR_YELLOW, 2, cv2.LINE_AA)
            # Texto de distancia
            import math
            dist = math.hypot(s_center[0] - c_center[0], s_center[1] - c_center[1])
            mid_x = (s_center[0] + c_center[0]) // 2
            mid_y = (s_center[1] + c_center[1]) // 2
            cv2.putText(image, f"{dist:.0f}px", (mid_x, mid_y - 5),
                       FONT, FONT_SMALL, COLOR_YELLOW, 1)

    return image
