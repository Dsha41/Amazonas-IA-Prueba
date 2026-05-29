"""Descarga el modelo YOLO de personas (yolo11m.pt) a models/base/.

Ultralytics descarga automaticamente el modelo desde el release oficial
de github si no existe localmente. Este script aprovecha eso y luego
mueve el archivo al directorio canonico del proyecto.

Uso:
    python scripts/setup_models.py
"""

import os
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MODELS_BASE = PROJECT_ROOT / "models" / "base"
YOLO_NAME = "yolo11m.pt"
YOLO_DEST = MODELS_BASE / YOLO_NAME


def step(m): print(f"\n[*] {m}")
def ok(m):   print(f"  [OK] {m}")
def err(m):  print(f"  [ERR] {m}")


def main() -> int:
    print("=" * 70)
    print(f"Setup: {YOLO_NAME} (YOLO COCO para deteccion de personas)")
    print("=" * 70)

    if YOLO_DEST.exists():
        size_mb = YOLO_DEST.stat().st_size / 1e6
        ok(f"Modelo ya existe: {YOLO_DEST} ({size_mb:.0f}MB)")
        return 0

    MODELS_BASE.mkdir(parents=True, exist_ok=True)

    step(f"Descargando {YOLO_NAME} via ultralytics")
    try:
        from ultralytics import YOLO
    except ImportError:
        err("ultralytics no instalado. Corre: pip install -r requirements.txt")
        return 1

    try:
        # Ultralytics descarga el .pt al cwd si no existe.
        model = YOLO(YOLO_NAME)
        # El path real del archivo descargado vive en model.ckpt_path o
        # se busca en el cwd con el nombre estandar.
        src_candidate = Path.cwd() / YOLO_NAME
        if not src_candidate.exists():
            # Fallback: ultralytics guarda en ~/.cache si cambio la api.
            for p in (Path.cwd(), Path.home() / ".cache" / "ultralytics"):
                cand = p / YOLO_NAME
                if cand.exists():
                    src_candidate = cand
                    break
        if not src_candidate.exists():
            err(f"No se encuentra {YOLO_NAME} tras la descarga")
            return 1

        shutil.move(str(src_candidate), str(YOLO_DEST))
        size_mb = YOLO_DEST.stat().st_size / 1e6
        ok(f"Movido a {YOLO_DEST} ({size_mb:.0f}MB)")
    except Exception as e:
        err(f"Descarga fallo: {e}")
        return 1

    print("=" * 70)
    ok("YOLO listo en models/base/.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
