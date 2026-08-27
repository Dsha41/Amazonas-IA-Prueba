"""
hummus_processor.py - Analitica de mostrador para el layout "Hummus".

Envuelve a PersonAmazonas en vez de modificarlo. El pipeline original
tiene ~3000 lineas y da servicio al layout que ya esta en produccion:
meterle ramas por tipo de layout significa que cada cambio de Hummus
arriesga romper "Personal de Amazonas". Envolviendolo, el objeto que
procesa sigue siendo exactamente el mismo que hoy funciona, y la logica
nueva vive en archivos que nadie mas importa.

Expone la misma interfaz que PersonAmazonas (delegando por __getattr__),
asi que app.py no distingue uno de otro.

QUE DETECTA
-----------
Dos eventos, con dos mecanismos distintos porque tienen formas distintas
en los datos:

  entrega de plato -> proximidad + cronometro (ZoneEventTracker)
      La entrega dibuja una V clara en la distancia empleado-cliente:
      se acercan, ocurre, se separan. El minimo coincide con el traspaso.

  toma de orden    -> permanencia en zona (ZoneDwellTracker)
      En la caja el cajero y el cliente estan a distancia casi constante
      durante medio minuto. No hay ningun instante que destaque, asi que
      la proximidad no puede encontrarlo; lo que si es evidente en los
      datos es que el cliente PERMANECE.

COMO SALEN LOS EVENTOS
----------------------
Por metadata["alerts"], el mismo canal que ya usa el servidor para las
alertas de pickup. El cliente (Amazonasview) lo consume de forma
generica en render_box.py: muestra el "event_type" que reciba, sin
tenerlo escrito a mano. Por eso esto no requiere ningun cambio en el
cliente.
"""

import base64
import datetime
import logging
import os
from typing import Any, Dict, List, Optional

import cv2

from .analytics.zone_event_tracker import ZoneEventTracker
from .analytics.zone_dwell_tracker import ZoneDwellTracker

logger = logging.getLogger(__name__)

# Coordenadas de zona sobre el frame de 960x576 de la camara "AM-CAJA1".
# Son especificas de ESA instalacion: otra camara necesita las suyas.
# Si existe hummus_zones.json en la raiz del proyecto, se usan las de ahi.
ZONAS_POR_DEFECTO: Dict[str, Any] = {
    "zona_caja": [[498, 364], [469, 514], [151, 435], [215, 308]],
    "zona_entrega": [[250, 182], [586, 278], [548, 361], [210, 255]],
    # Todo el personal (sirve para no contar a un empleado como cliente).
    "lado_staff": [[588, -1], [746, 3], [809, 591], [226, 569]],
    # Solo quien SIRVE los platos. La cajera queda fuera a proposito: esta
    # en el centro del encuadre, asi que quedaba cerca de todo el mundo y
    # disparaba entregas que nunca ocurrieron. El corte salio de los datos
    # -- los cajeros se mueven entre y=385 y y=506, los servidores entre
    # y=63 y y=360, y nadie cruza y~370.
    "zona_servidor": [[588, -1], [746, 3], [785, 370], [352, 370]],
    # El mostrador impone un suelo fisico de ~197 px entre empleado y
    # cliente. Pedir menos es pedir algo imposible; 250 px equivale a
    # "a un mostrador de distancia".
    "distance_px": 250.0,
    "min_time_sec": 1.0,
    # Sin tolerancia, un solo frame fallido reinicia el cronometro. Medido
    # sobre el video: una oclusion de 0.6 s retrasaba una entrega 2 s.
    "max_gap_sec": 1.0,
    "orden_dwell_sec": 5.0,
    "orden_gap_sec": 1.0,
}

ZONES_FILE = "hummus_zones.json"


def cargar_zonas(path: str = ZONES_FILE) -> Dict[str, Any]:
    """Lee las zonas de un JSON si existe; si no, usa las de por defecto."""
    cfg = dict(ZONAS_POR_DEFECTO)
    if not os.path.exists(path):
        logger.info("Hummus: %s no existe, usando zonas por defecto", path)
        return cfg

    try:
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Hummus: no se pudo leer %s (%s). Usando zonas por "
                       "defecto.", path, e)
        return cfg

    for clave in ("zona_caja", "zona_entrega", "lado_staff", "zona_servidor"):
        pts = data.get(clave)
        if pts and len(pts) >= 3:
            cfg[clave] = pts
        elif pts:
            logger.warning("Hummus: '%s' tiene %d punto(s); un poligono "
                           "necesita 3. Se ignora.", clave, len(pts))
    for clave in ("distance_px", "min_time_sec", "max_gap_sec",
                  "orden_dwell_sec", "orden_gap_sec"):
        if data.get(clave) is not None:
            try:
                cfg[clave] = float(data[clave])
            except (TypeError, ValueError):
                logger.warning("Hummus: '%s' no es un numero. Se ignora.", clave)

    logger.info("Hummus: zonas cargadas de %s", path)
    return cfg


class HummusProcessor:
    """Compone un PersonAmazonas y le anade los eventos de mostrador."""

    # Como se anuncia cada evento en el panel de alertas del cliente.
    ETIQUETAS = {
        "toma_de_orden": "Toma de orden",
        "entrega_de_plato": "Entrega de plato",
    }

    def __init__(self, inner, zonas: Optional[Dict[str, Any]] = None,
                 output_dirs: Optional[Dict[str, str]] = None):
        self.inner = inner
        self.zonas = zonas if zonas is not None else cargar_zonas()
        self._dirs = output_dirs or {}

        # Un juego de detectores por camara: dos camaras no deben compartir
        # el cronometro de una entrega.
        self._por_camara: Dict[Any, "HummusProcessor"] = {}

        z = self.zonas
        self._entrega = ZoneEventTracker(
            event_name="entrega_de_plato",
            zone_polygon=z["zona_entrega"],
            staff_side_polygon=z["lado_staff"],
            staff_pair_polygon=z.get("zona_servidor"),
            distance_px=z["distance_px"],
            min_time_sec=z["min_time_sec"],
            max_gap_sec=z["max_gap_sec"],
        )
        self._orden = ZoneDwellTracker(
            event_name="toma_de_orden",
            zone_polygon=z["zona_caja"],
            staff_side_polygon=z["lado_staff"],
            min_dwell_sec=z["orden_dwell_sec"],
            max_gap_sec=z["orden_gap_sec"],
        )

    # ── Interfaz de PersonAmazonas ──────────────────────────────────

    def get_camera_processor(self, camera_id: Any):
        """Devuelve el procesador de una camara, TAMBIEN envuelto.

        app.py hace processor.get_camera_processor(cam) y llama a
        process_frame() sobre el resultado. Si aqui devolvieramos el
        PersonAmazonas de dentro sin envolver, el frame se procesaria
        saltandose por completo la analitica de Hummus -- sin dar ningun
        error, simplemente no saldrian eventos.
        """
        if camera_id not in self._por_camara:
            self._por_camara[camera_id] = HummusProcessor(
                self.inner.get_camera_processor(camera_id),
                zonas=self.zonas,
                output_dirs=self._dirs,
            )
        return self._por_camara[camera_id]

    def process_frame(self, image, *args, **kwargs):
        """Procesa el frame y anade los eventos de mostrador a la metadata."""
        resultado = self.inner.process_frame(image, *args, **kwargs)

        # process_frame devuelve (imagen, metadata), salvo que algo haya
        # fallado dentro y devuelva un dict de error.
        if not (isinstance(resultado, tuple) and len(resultado) == 2):
            return resultado
        img, metadata = resultado
        if not isinstance(metadata, dict):
            return resultado

        camera_id = kwargs.get("camera_id")
        if camera_id is None and len(args) >= 3:
            camera_id = args[2]      # process_frame(img, roi, activate, camera_id)
        if camera_id is None:
            camera_id = 1

        try:
            tracks = self.inner.get_active_tracks(camera_id)
            if tracks:
                nuevos: List[Dict[str, Any]] = []
                nuevos += self._entrega.update(tracks)
                nuevos += self._orden.update(tracks)
                for ev in nuevos:
                    metadata.setdefault("alerts", []).append(
                        self._construir_alerta(ev, img)
                    )
        except Exception as e:
            # Un fallo aqui no debe tumbar el frame: la imagen procesada
            # y el resto de la metadata siguen siendo validas.
            logger.error("Hummus: error generando eventos: %s", e)

        return img, metadata

    def __getattr__(self, name):
        """Todo lo que no conocemos se delega al procesador de dentro."""
        # __getattr__ solo se llama si el atributo NO se encontro por la
        # via normal, asi que no intercepta lo definido arriba.
        return getattr(self.inner, name)

    # ── Construccion de la alerta ───────────────────────────────────

    def _construir_alerta(self, ev: Dict[str, Any], img) -> Dict[str, Any]:
        """Da a un evento el formato que el cliente espera en 'alerts'."""
        nombre = ev.get("event", "")
        etiqueta = self.ETIQUETAS.get(nombre, nombre or "Alerta")
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if nombre == "entrega_de_plato":
            desc = (f"Servidor {ev.get('staff_id')} -> cliente "
                    f"{ev.get('client_id')} ({ev.get('duration', 0):.1f} s)")
        else:
            desc = (f"Cliente {ev.get('client_id')} permanecio "
                    f"{ev.get('duration', 0):.1f} s en la caja")

        img_b64 = self._a_base64(img)
        ruta = self._guardar_captura(nombre, img)

        return {
            "event_type": etiqueta,
            "class_name": nombre,
            "description": desc,
            "timestamp": ts,
            "image_base64": img_b64,
            "crop_image": img_b64,
            "screenshot_path": ruta,
            # Extras: el cliente los ignora, pero quedan en el log del
            # servidor y sirven para depurar.
            "client_track_id": ev.get("client_id"),
            "staff_track_id": ev.get("staff_id"),
            "event_start_sec": ev.get("start_time"),
        }

    @staticmethod
    def _a_base64(img) -> str:
        try:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                return ""
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as e:
            logger.debug("Hummus: no se pudo codificar la captura: %s", e)
            return ""

    def _guardar_captura(self, nombre: str, img) -> str:
        """Guarda la captura en el directorio que ya define config.py.

        config.py declara hummus_order_screenshot_dir y
        hummus_delivery_screenshot_dir apuntando a Toma_de_orden_hummus/ y
        Entrega_de_plato_hummus/. Los autores originales tenian pensado
        exactamente esto, asi que se respetan esas rutas.
        """
        clave = ("hummus_delivery_screenshot_dir"
                 if nombre == "entrega_de_plato"
                 else "hummus_order_screenshot_dir")
        carpeta = self._dirs.get(clave)
        if not carpeta:
            return ""
        try:
            os.makedirs(carpeta, exist_ok=True)
            marca = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            ruta = os.path.join(carpeta, f"{nombre}_{marca}.jpg")
            cv2.imwrite(ruta, img)
            return ruta
        except Exception as e:
            logger.warning("Hummus: no se pudo guardar la captura: %s", e)
            return ""

    # ── Estado ──────────────────────────────────────────────────────

    def get_hummus_stats(self) -> Dict[str, Any]:
        return {
            "entrega": self._entrega.get_stats(),
            "orden": self._orden.get_stats(),
        }

    def reset_hummus(self):
        self._entrega.reset()
        self._orden.reset()
        for sub in self._por_camara.values():
            sub.reset_hummus()
