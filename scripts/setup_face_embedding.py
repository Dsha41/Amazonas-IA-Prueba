"""Descarga el modelo ArcFace w600k_r50.onnx para face re-identification.

Es el modelo de embeddings 512-dim que usaremos para evitar contar la
misma persona dos veces. Viene en el pack buffalo_l de InsightFace.

Uso:
  python scripts/setup_face_embedding.py
"""

import os
import sys
import urllib.request
import zipfile
import shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'classifiers')
TMP_DIR = os.path.join(PROJECT_ROOT, 'tmp_buffalo_l')
ZIP_URL = ("https://github.com/deepinsight/insightface/releases/"
           "download/v0.7/buffalo_l.zip")
ZIP_PATH = os.path.join(TMP_DIR, 'buffalo_l.zip')
EMBED_OUT = os.path.join(MODELS_DIR, 'w600k_r50.onnx')


def step(m): print(f"\n[*] {m}")
def ok(m): print(f"  [OK] {m}")
def err(m): print(f"  [ERR] {m}")


def download_with_progress(url, dest):
    last_pct = [-1]
    def report(bn, bs, ts):
        d = bn * bs
        pct = int(d * 100 / ts) if ts > 0 else 0
        if pct != last_pct[0] and pct % 10 == 0:
            print(f"  {pct}% ({d/1e6:.0f}/{ts/1e6:.0f} MB)")
            last_pct[0] = pct
    urllib.request.urlretrieve(url, dest, reporthook=report)


def main():
    print("=" * 70)
    print("Setup: w600k_r50.onnx (ArcFace embedding model)")
    print("=" * 70)

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.isfile(EMBED_OUT):
        size_mb = os.path.getsize(EMBED_OUT) / 1e6
        ok(f"Modelo ya existe: {EMBED_OUT} ({size_mb:.0f}MB)")
        return

    step("Descargando buffalo_l.zip (~280MB) si no esta")
    if os.path.isfile(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 100_000_000:
        ok(f"Zip ya existe ({os.path.getsize(ZIP_PATH)/1e6:.0f}MB)")
    else:
        try:
            download_with_progress(ZIP_URL, ZIP_PATH)
            ok(f"Descargado: {os.path.getsize(ZIP_PATH)/1e6:.0f}MB")
        except Exception as e:
            err(f"Descarga fallo: {e}")
            sys.exit(1)

    step("Extrayendo w600k_r50.onnx")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        members = zf.namelist()
        target = None
        for m in members:
            if 'w600k_r50' in m.lower() and m.endswith('.onnx'):
                target = m
                break
        if target is None:
            err("No se encontro w600k_r50.onnx en el zip")
            sys.exit(1)
        with zf.open(target) as src, open(EMBED_OUT, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    size_mb = os.path.getsize(EMBED_OUT) / 1e6
    ok(f"Extraido: {EMBED_OUT} ({size_mb:.0f}MB)")

    step("Verificando")
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(
            EMBED_OUT, providers=['CPUExecutionProvider']
        )
        ins = sess.get_inputs()
        outs = sess.get_outputs()
        print(f"  Inputs:  {[(i.name, i.shape) for i in ins]}")
        print(f"  Outputs: {[(o.name, o.shape) for o in outs]}")
        # Test forward (input es 112x112 RGB normalizado)
        dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
        res = sess.run([outs[0].name], {ins[0].name: dummy})
        print(f"  Output shape: {res[0].shape}")
        ok("Modelo funcional")
    except Exception as e:
        err(f"Verificacion fallo: {e}")
        sys.exit(1)

    print(f"\n[*] Limpieza: Remove-Item -Recurse -Force \"{TMP_DIR}\"")
    print("=" * 70)
    ok("ArcFace listo. Face re-identification activable.")
    print("=" * 70)


if __name__ == "__main__":
    main()
