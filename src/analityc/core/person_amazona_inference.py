import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from collections import defaultdict, deque, Counter
import logging
from typing import Tuple, Dict, Any, List
import asyncio
import threading
import base64
import httpx
import datetime
import os
import traceback
from ..config.config import DEFAULT_ROI

# ── Modulos de analitica retail ──
from .analytics.config import AnalyticsConfig
from .analytics.demographics import DemographicsClassifier
from .analytics.face_reidentifier import FaceReidentifier
from .analytics.people_counter import PeopleCounter
from .analytics.attendance_tracker import AttendanceTracker
from .analytics.seller_efficiency import SellerEfficiency
from .analytics.stock_monitor import StockMonitor
from .utils.overlay import (
    draw_demographics_panel, draw_people_total, draw_seller_dashboard,
    draw_award_banner, draw_stock_indicators, draw_proximity_lines,
    draw_products_total,
)
from .utils.logger import AnalyticsLogger

logger = logging.getLogger(__name__)

# ── Constantes para el clasificador de género / edad (Caffe DNN) ──────
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'models', 'classifiers')
_GENDER_PROTO = os.path.join(_MODELS_DIR, 'gender_deploy.prototxt')
_GENDER_MODEL = os.path.join(_MODELS_DIR, 'gender_net.caffemodel')
_AGE_PROTO    = os.path.join(_MODELS_DIR, 'age_deploy.prototxt')
_AGE_MODEL    = os.path.join(_MODELS_DIR, 'age_net.caffemodel')
# ── InsightFace genderage.onnx (primario, mas preciso) ──────────────────
_INSIGHTFACE_GENDERAGE = os.path.join(_MODELS_DIR, 'genderage.onnx')
# ── MiVOLO ONNX (modelo SOTA 2024 opcional para ensemble) ─────────────
# Si esta presente activa ensemble con acuerdo de genero. Descargar de
# https://github.com/WildChlamydia/MiVOLO, convertir a ONNX y copiar a
# models/classifiers/mivolo.onnx
_MIVOLO_MODEL = os.path.join(_MODELS_DIR, 'mivolo.onnx')
# ── ArcFace w600k_r50 ONNX (Face Re-Identification) ──────────────────
# Modelo de embeddings 512-dim para evitar contar la misma persona dos
# veces. Se descarga con scripts/setup_face_embedding.py
_FACE_EMBEDDING_MODEL = os.path.join(_MODELS_DIR, 'w600k_r50.onnx')
# YuNet face detector (2023 mar): detector moderno con keypoints
# (ojos, nariz, boca) que permite alineacion facial. SOTA para
# caras pequenas/ladeadas en CCTV.
_YUNET_MODEL = os.path.join(_MODELS_DIR, 'face_detection_yunet_2023mar.onnx')


class PersonAmazonas:
    """Procesador para reconocimiento de personal específico"""
    
    # Rutas por defecto resueltas relativas al root del proyecto
    # (sube 4 niveles desde este archivo: core/ -> analityc/ -> src/ -> root/)
    _PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', '..', '..')
    )
    # YOLO estándar COCO para detección de personas
    _DEFAULT_PERSON_MODEL = os.path.join(
        _PROJECT_ROOT, 'models', 'base', 'yolo11m.pt'
    )
    # Modelo de productos cosmeticos (16 SKUs retail). Opcional: solo se
    # carga si cosmetics_enabled=True. Por defecto el toggle está OFF.
    _DEFAULT_PRODUCT_MODEL = os.path.join(
        _PROJECT_ROOT, 'models', 'cosmeticos', 'weights', 'best.pt'
    )

    def __init__(self,
                client_id: None = None,
                model_path: str = None,  # Modelo de productos (cosmeticos best.pt)
                person_model_path: str = None,     # Modelo YOLO para personas
                confidence_threshold: float = 0.7,
                iou_threshold: float = 0.4,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                log_file: str = "output/detection_log.txt",
                image_quality: int = 70,
                min_time_in_roi: int = 10,
                max_frames_out: int = 5,
                min_track_frames: int = 3,
                show_minimal_info: bool = True,
                exit_frames_threshold: int = 1,
                max_frames_without_detection: int = 5,
                max_image_size: tuple = (640, 480),
                staff_names_file: str = None,
                shared_model: Any = None,
                shared_person_model: Any = None):
        
        try:
            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold
            self.model_path = model_path or self._DEFAULT_PRODUCT_MODEL
            self.person_model_path = person_model_path or self._DEFAULT_PERSON_MODEL
            self.log_file = log_file
            self.image_quality = image_quality
            self.show_minimal_info = show_minimal_info
            self.exit_frames_threshold = exit_frames_threshold
            self.max_image_size = max_image_size
            
            # Configuración de tiempos
            self.min_time_in_roi = min_time_in_roi
            self.max_frames_out = max_frames_out
            self.min_track_frames = min_track_frames
            self.max_frames_without_detection = max_frames_without_detection
            
            # Estado interno
            self.frame_counter = 0
            self.personas_en_area = 0
            self.active_tracks = {}
            self.next_id = 1

            # Toggle cliente: si False se omite TODA la inferencia del modelo
            # de cosmeticos (model 1) y los pases auxiliares de SKU. La
            # deteccion de personas + genero + edad sigue funcionando.
            # Cambiado a False: enfoque exclusivo en personas/demograficos.
            self.cosmetics_enabled = False
            self.track_history = defaultdict(lambda: deque(maxlen=30))
            
            # Contadores por empleado
            self.employee_counters = defaultdict(int)

            # Contadores de entradas por categoría
            self.entry_counts = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}
            
            # ROI por defecto
            self.roi_polygon = np.array(DEFAULT_ROI if DEFAULT_ROI else [(0, 0), (640, 0), (640, 480), (0, 480)], np.int32)
            
            # Para evitar reconteo
            self.counted_tracks = set()
            self.recent_counted_persons = deque(maxlen=30)
            self.person_cooldown = defaultdict(int)
            
            # Estados de seguimiento
            self.last_counted_frame = 0
            self.last_counted_id = 0
            self.debug_mode = True
            
            # Historial de posiciones para validar movimiento
            self.movement_history = defaultdict(lambda: deque(maxlen=10))

            # Historial de clases por track para estabilizar identidad
            self.class_history_len = 8
            self.class_stability_threshold = 0.6  # proporción para considerar estable
            self.track_class_history = defaultdict(lambda: deque(maxlen=self.class_history_len))
            
            # Nombres del personal (se cargan desde archivo o modelo)
            self.staff_names = {}
            self.load_staff_names(staff_names_file)
            
            # Clases a detectar - SOLO PERSONAL
            self.all_classes = []  # Se determinará después de cargar el modelo
            
            # Para controlar alertas periódicas
            self.alert_minutes_sent = defaultdict(list)
            
            # Para controlar que no se envíen múltiples fotos del mismo evento
            self.sent_entry_photos = defaultdict(lambda: deque(maxlen=2))
            self.sent_exit_photos = defaultdict(lambda: deque(maxlen=2))
            
            # Control de frecuencia de envío
            self.last_sent_time = defaultdict(float)
            self.send_cooldown = 1.0
            
            self.model = None
            self.person_model = None
            self.device = device
            # Identificador del cliente/instancia
            self.client_id = client_id
            print(device)
            # Si se proporciona un modelo compartido, usarlo y evitar
            # re-inicializar. El shared_model puede ser None cuando
            # cosmetics_enabled=False (no se carga el modelo).
            if shared_person_model is not None:
                # Caso compartido: la instancia padre ya cargo los modelos
                self.model = shared_model  # puede ser None
                self.person_model = shared_person_model
                if (self.model is not None and
                        hasattr(self.model, 'names') and self.model.names):
                    self.all_classes = list(self.model.names.keys())
                    self.staff_names = {
                        cid: self.model.names[cid]
                            .replace('_', ' ').title()
                        for cid in self.all_classes
                    }
                else:
                    self.all_classes = []
                    self.staff_names = {}
                self.staff_names[-1] = 'Persona'
            else:
                self._initialize_model()

            # Cache de procesadores por cámara (si se usa una instancia compartida)
            self._camera_processors = {}

            # Estado por cámara (separado)
            # Cada camera_id tendrá su propio conjunto de tracks y contadores
            # camera_states[camera_id] = {
            #    'active_tracks': {}, 'next_id': int, 'track_history': defaultdict(deque),
            #    'movement_history': defaultdict(deque), 'employee_counters': defaultdict(int), ...
            # }
            self.camera_states = {}
            self._state_swap_stack = []
            # Lock to prevent concurrent process_frame calls interfering with state swapping
            self._process_lock = threading.RLock()
            # atributo auxiliar para referencia al último frame procesado
            self.last_processed_frame = None
            
            os.makedirs("output/Personal", exist_ok=True)
            os.makedirs("output/Personal/Entradas", exist_ok=True)
            os.makedirs("output/Personal/Salidas", exist_ok=True)
            os.makedirs("output/Personal/Alertas", exist_ok=True)
            os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
            
            self._log_buffer = []
            self.setup_log_file()

            # ── Clasificador de género / edad (OpenCV DNN) ──
            self._init_gender_age_classifier()
            # Intervalo: clasificar cada frame mientras el track aun no
            # haya convergido. Con el nuevo voting robusto se necesitan
            # 6 muestras de calidad para commit, asi que queremos llenarlas
            # lo mas rapido posible y ademas descartar las ruidosas.
            self._classify_every_n = 1
            # Umbral de altura relativa para niño (fallback sin modelo de edad)
            self.child_height_ratio = 0.30

            # ── Modulos de analitica retail ──
            self._analytics_config = AnalyticsConfig
            # Face Re-Identification (compartido entre demographics y este)
            self._reidentifier = FaceReidentifier(
                embedding_session=getattr(self, '_face_embedding_session',
                                          None),
                db_path=AnalyticsConfig.REID_DB_PATH,
                similarity_threshold=(
                    AnalyticsConfig.REID_SIMILARITY_THRESHOLD),
                reset_policy=AnalyticsConfig.REID_RESET_POLICY,
            )
            # Guardar la DB al cerrar el proceso (Ctrl+C / shutdown)
            try:
                import atexit
                atexit.register(self._reidentifier.force_save)
            except Exception:
                pass

            self._demographics = DemographicsClassifier(
                gender_net=self._gender_net,
                age_net=self._age_net,
                face_cascade=self._face_cascade,
                face_detector_dnn=getattr(self, '_face_dnn', None),
                insightface_session=getattr(self, '_insightface_session', None),
                yunet=getattr(self, '_yunet', None),
                mivolo_session=getattr(self, '_mivolo_session', None),
                reidentifier=self._reidentifier,
            )
            self._people_counter = PeopleCounter()
            # Contador de productos cosmeticos (separado de personas)
            # _product_track_ids: set de track_ids de cosmeticos ya vistos.
            # _product_counts_by_sku: {nombre_sku: cantidad_de_tracks_unicos}
            self._product_track_ids: set = set()
            self._product_counts_by_sku: Dict[str, int] = {}
            # Eventos pickup: pares (person_tid, sku_tid) ya emitidos para
            # no repetir alertas. Y cooldown por persona+sku en segundos.
            self._pickup_emitted: set = set()
            self._pickup_last_emit: Dict[tuple, float] = {}
            # Buffer de alertas pickup pendientes para incluir en metadata
            self._pending_pickup_alerts: List[Dict[str, Any]] = []
            # SKUs actualmente "en mano": el track no expira hasta que el
            # producto vuelve al estante o el holder desaparece.
            # _held_skus[sku_tid] = {
            #     'person_tid': int, 'pickup_frame': int, 'pickup_time': float,
            #     'last_box': np.array, 'sku_class': int,
            #     'frames_back_on_shelf': int, 'released_emitted': bool
            # }
            self._held_skus: Dict[int, Dict[str, Any]] = {}
            # Directorio para guardar screenshots de pickup
            self._pickup_dir = os.path.join(
                os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.',
                "pickups"
            )
            try:
                os.makedirs(self._pickup_dir, exist_ok=True)
            except Exception:
                pass
            self._attendance_tracker = AttendanceTracker()
            self._seller_efficiency = SellerEfficiency()
            self._stock_monitor = StockMonitor()
            self._analytics_logger = AnalyticsLogger()

            print(f'✅ Modelo de personal inicializado para {client_id}')
            print(f'🖥️  Dispositivo: {self.device}')
            print(f'🎯 Umbral de confianza: {confidence_threshold}')
            print(f'👥 Personal registrado: {len(self.staff_names)} personas')
            print(f'📍 Modo debug: {"ACTIVADO" if self.debug_mode else "DESACTIVADO"}')
        except Exception as e:
            # NO tragar la excepcion: imprimir traza completa y re-lanzar
            # para que el factory en app.py sepa que la instancia no quedo usable.
            print(f"❌ Error fatal en PersonAmazonas.__init__: {e}")
            print(traceback.format_exc())
            logger.error("PersonAmazonas.__init__ fallo: %s", e, exc_info=True)
            raise



    def load_staff_names(self, staff_names_file: str = None):
        """Carga los nombres del personal desde archivo o modelo"""
        try:
            if staff_names_file and os.path.exists(staff_names_file):
                with open(staff_names_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            class_id, name = line.strip().split(':')
                            self.staff_names[int(class_id)] = name
                print(f"📋 Nombres cargados desde archivo: {len(self.staff_names)} empleados")
            else:
                print("ℹ️  No se encontró archivo de nombres, se usarán los del modelo")
        except Exception as e:
            print(f"⚠️  Error cargando nombres: {e}")

    # Lista estatica de clases esperadas del modelo cosmeticos (referencia).
    # En runtime se lee self.model.names y se activan TODAS las clases.
    _EXPECTED_COSMETIC_CLASSES = [
        'ACEITE_ARGAN_FORRELLE',
        'ACEITE_HIDRATANTE_ALMENDRA_BRIZNA',
        'ACEITE_HIDRATANTE_ARGAN_BRIZNA',
        'ACEITE_HIDRATANTE_COCO_BRIZNA',
        'ACEITE_HIDRATANTE_ROSA_MOSQUETA',
        'ACONDICIONADO_ACEITE_ARGAN_FORELLE',
        'AMPOLLA_NUTRITIVO_ARGAN_FORELLE',
        'CHAMPU_ANTICAIDA_KAP',
        'CHAMPU_ANTICASPA_KAP',
        'CREMA_PEINAR_MOLDEADORA_FORELLE',
        'LACA_FIJADORA_GELLY',
        'LACA_LIQUIDA_ARGAN_FORELLE',
        'LOCION_CAPOLAR_VITAMINAE_BRIZNA',
        'RESTAURACION_EXTREMA_FORELLE',
        'SOLANO_LOCION_NUTRITIVA_EFECTO_BOTOX',
        'SPLASH_CORPORAL_MANGO_ALOE_BRIZNA',
    ]
    # Compat: algunos caminos de codigo antiguos siguen consultando
    # _ALLOWED_CLASS_IDS. Se calcula dinamicamente tras cargar el modelo.
    _ALLOWED_CLASS_IDS = []

    def _initialize_model(self):
        try:
            # ── Modelo 1: productos cosmeticos (OPCIONAL) ──
            # Solo se carga si cosmetics_enabled. El enfoque del sistema
            # es deteccion de personas + genero + edad; cosmeticos es
            # secundario y consume GPU/CPU innecesariamente.
            if self.cosmetics_enabled:
                print(f"🚀 Inicializando modelo cosmeticos: "
                      f"{self.model_path}")
                self.model = YOLO(self.model_path).to(self.device)

                if hasattr(self.model, 'names') and self.model.names:
                    print(f"✅ Clases disponibles en cosmeticos: "
                          f"{self.model.names}")
                    self.all_classes = list(self.model.names.keys())
                    self.staff_names = {
                        cid: self.model.names[cid]
                            .replace('_', ' ').title()
                        for cid in self.all_classes
                    }
                    type(self)._ALLOWED_CLASS_IDS = list(self.all_classes)
                else:
                    self.all_classes = list(range(
                        len(self._EXPECTED_COSMETIC_CLASSES)))
                    self.staff_names = {
                        i: name.replace('_', ' ').title()
                        for i, name in enumerate(
                            self._EXPECTED_COSMETIC_CLASSES)
                    }
            else:
                print("ℹ️  Modelo de cosmeticos DESACTIVADO "
                      "(cosmetics_enabled=False). Enfoque exclusivo en "
                      "personas y demograficos.")
                self.model = None
                self.all_classes = []
                self.staff_names = {}
                type(self)._ALLOWED_CLASS_IDS = []

            # ── Modelo 2: YOLO estándar (personas COCO clase 0) ──
            print(f"🚀 Inicializando modelo personas: "
                  f"{self.person_model_path}")
            self.person_model = YOLO(self.person_model_path).to(
                self.device)
            # Usamos class_id -1 internamente para personas
            self.staff_names[-1] = 'Persona'

            print(f"✅ Detectando personas con: {self.person_model_path}")

            # Calentamiento del modelo de personas (siempre)
            dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
            if self.model is not None:
                _ = self.model.predict(
                    dummy_input, imgsz=320, device=self.device,
                    classes=self.all_classes, verbose=False
                )
            _ = self.person_model.predict(
                dummy_input, imgsz=320, device=self.device,
                classes=[0], verbose=False
            )
            print(f"✅ Modelos inicializados "
                  f"(cosmeticos={'ON' if self.model else 'OFF'})")
        except Exception as e:
            print(f"❌ Error inicializando modelos: {e}")
            raise

    def setup_log_file(self):
        try:
            with open(self.log_file, 'w', encoding="utf-8") as f:
                f.write("Timestamp,Frame,Empleado_ID,Empleado_Nombre,Evento,Tiempo_Area,Confianza\n")
        except Exception as e:
            print(f"❌ Error creando archivo de log: {e}")

    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        x11, y11, x21, y21 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x11, x1_2)
        yi1 = max(y11, y1_2)
        xi2 = min(x21, x2_2)
        yi2 = min(y21, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def center_of(self, box: Tuple) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def is_inside_polygon(self, point: Tuple, polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

    def compress_image(self, image: np.ndarray) -> np.ndarray:
        try:
            height, width = image.shape[:2]
            max_width, max_height = self.max_image_size
            
            if width > max_width or height > max_height:
                aspect_ratio = width / height
                if width > max_width:
                    width = max_width
                    height = int(width / aspect_ratio)
                if height > max_height:
                    height = max_height
                    width = int(height * aspect_ratio)
                
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            
            return image
        except Exception as e:
            logger.error(f"Error al comprimir imagen: {e}")
            return image

    def get_staff_name(self, class_id: int, confidence: float = None) -> str:
        """Obtiene el nombre legible del empleado"""
        if class_id in self.staff_names:
            name = self.staff_names[class_id]
            if confidence:
                return f"{name} ({confidence:.2f})"
            return name
        else:
            if confidence:
                return f"Empleado_{class_id} ({confidence:.2f})"
            return f"Empleado_{class_id}"

    def get_staff_display_name(self, class_id: int) -> str:
        """Nombre para mostrar en pantalla"""
        if class_id in self.staff_names:
            return self.staff_names[class_id]
        return f"ID_{class_id}"

    # ── Clasificación por categoría (Hombre / Mujer / Niño) ──────
    _CATEGORY_KEYWORDS = {
        'Hombres': ['hombre', 'hombres', 'man', 'male', 'boy', 'chico'],
        'Mujeres': ['mujer', 'mujeres', 'woman', 'female', 'girl', 'chica'],
        'Niños':   ['niño', 'niña', 'niños', 'child', 'kid', 'infant', 'bebe', 'baby'],
    }

    def _classify_category(self, class_id: int) -> str:
        """Clasifica un class_id del modelo en Hombres/Mujeres/Niños/Personas."""
        name = self.get_staff_display_name(class_id).lower().strip()
        for category, keywords in self._CATEGORY_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                return category
        return 'Personas'

    # ── Clasificador de género / edad con OpenCV DNN ──────────────

    def _init_gender_age_classifier(self):
        """Carga los modelos Caffe de género y edad. Si faltan, desactiva.

        Adicionalmente, intenta cargar el modelo InsightFace genderage.onnx
        como PRIMARIO (mucho mas preciso que el Caffe Levi-Hassner 2015).
        Si ONNX no esta disponible, cae a Caffe automaticamente.
        """
        self._gender_net = None
        self._age_net = None
        self._face_cascade = None
        self._face_dnn = None
        self._yunet = None
        self._insightface_session = None
        self._mivolo_session = None
        self._face_embedding_session = None

        try:
            if os.path.exists(_GENDER_MODEL) and os.path.exists(_GENDER_PROTO):
                self._gender_net = cv2.dnn.readNet(_GENDER_MODEL, _GENDER_PROTO)
                logger.info("Gender classifier loaded")
            else:
                logger.warning(f"Gender model not found in {_MODELS_DIR}")

            if os.path.exists(_AGE_MODEL) and os.path.exists(_AGE_PROTO):
                self._age_net = cv2.dnn.readNet(_AGE_MODEL, _AGE_PROTO)
                logger.info("Age classifier loaded")
            else:
                logger.warning(f"Age model not found in {_MODELS_DIR}")

            # YuNet face detector (primario moderno, devuelve 5 keypoints)
            if os.path.exists(_YUNET_MODEL):
                try:
                    self._yunet = cv2.FaceDetectorYN_create(
                        _YUNET_MODEL,
                        "",
                        (320, 320),
                        0.55,   # score threshold
                        0.30,   # nms threshold
                        5000,   # top_k
                    )
                    logger.info("YuNet face detector loaded (keypoints OK)")
                except Exception as e:
                    logger.warning(f"No se pudo cargar YuNet: {e}")
                    self._yunet = None

            # Face detector DNN (Res10 SSD) como fallback secundario
            face_proto = os.path.join(_MODELS_DIR, 'opencv_face_detector.pbtxt')
            face_model = os.path.join(_MODELS_DIR, 'opencv_face_detector_uint8.pb')
            if os.path.exists(face_proto) and os.path.exists(face_model):
                try:
                    self._face_dnn = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
                    logger.info("Face detector DNN loaded (Res10 SSD)")
                except Exception as e:
                    logger.warning(f"No se pudo cargar face DNN: {e}")
                    self._face_dnn = None

            # InsightFace genderage.onnx (primario, preciso >95%)
            if os.path.exists(_INSIGHTFACE_GENDERAGE):
                try:
                    import onnxruntime as ort
                    providers = ['CPUExecutionProvider']
                    if torch.cuda.is_available():
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self._insightface_session = ort.InferenceSession(
                        _INSIGHTFACE_GENDERAGE, providers=providers
                    )
                    logger.info("InsightFace genderage ONNX loaded")
                except Exception as e:
                    logger.warning(f"No se pudo cargar InsightFace ONNX: {e}")
                    self._insightface_session = None
            else:
                logger.info("InsightFace genderage.onnx no presente, "
                            "usando Caffe como primario")

            # ArcFace w600k_r50 ONNX (Face Re-Identification)
            if (AnalyticsConfig.REID_ENABLED and
                    os.path.exists(_FACE_EMBEDDING_MODEL)):
                try:
                    import onnxruntime as ort
                    providers = ['CPUExecutionProvider']
                    if torch.cuda.is_available():
                        providers = ['CUDAExecutionProvider',
                                     'CPUExecutionProvider']
                    self._face_embedding_session = ort.InferenceSession(
                        _FACE_EMBEDDING_MODEL, providers=providers
                    )
                    logger.info(
                        "ArcFace w600k_r50 loaded -> "
                        "Face Re-Identification activa"
                    )
                except Exception as e:
                    logger.warning(
                        f"No se pudo cargar ArcFace embedding: {e}"
                    )
                    self._face_embedding_session = None
            elif not AnalyticsConfig.REID_ENABLED:
                logger.info("Face Re-Identification deshabilitada en config")
            else:
                logger.info(
                    "ArcFace no presente. Descargar con: "
                    "python scripts/setup_face_embedding.py"
                )

            # MiVOLO ONNX (SOTA 2024, opcional para ensemble)
            if os.path.exists(_MIVOLO_MODEL):
                try:
                    import onnxruntime as ort
                    providers = ['CPUExecutionProvider']
                    if torch.cuda.is_available():
                        providers = ['CUDAExecutionProvider',
                                     'CPUExecutionProvider']
                    self._mivolo_session = ort.InferenceSession(
                        _MIVOLO_MODEL, providers=providers
                    )
                    logger.info("MiVOLO ONNX loaded -> ensemble activado")
                except Exception as e:
                    logger.warning(f"No se pudo cargar MiVOLO ONNX: {e}")
                    self._mivolo_session = None
            else:
                logger.info(
                    "MiVOLO no presente en %s. Para activar ensemble "
                    "SOTA descargar de https://github.com/WildChlamydia/MiVOLO",
                    _MIVOLO_MODEL
                )

            # Haar cascade (fallback y default)
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            has_gender = self._gender_net is not None
            has_age = self._age_net is not None
            has_face_dnn = self._face_dnn is not None
            has_yunet = self._yunet is not None
            has_insight = self._insightface_session is not None
            has_mivolo = self._mivolo_session is not None
            ensemble_n = sum([has_insight, has_mivolo, has_gender])
            primary = "MiVOLO-ONNX" if has_mivolo else (
                "InsightFace-ONNX" if has_insight else (
                    "Caffe" if has_gender else "NONE"
                )
            )
            detector = "YuNet+keypoints+pose" if has_yunet else (
                "Res10-SSD" if has_face_dnn else "Haar"
            )
            has_reid = self._face_embedding_session is not None
            print(f"🧑 Clasificador primario: {primary} | "
                  f"Ensemble: {ensemble_n} modelos | "
                  f"Detector: {detector} | "
                  f"MiVOLO: {'OK' if has_mivolo else 'NO'} | "
                  f"InsightFace: {'OK' if has_insight else 'NO'} | "
                  f"Caffe género: {'OK' if has_gender else 'NO'} | "
                  f"Caffe edad: {'OK' if has_age else 'NO'} | "
                  f"Re-ID biométrico: {'OK' if has_reid else 'NO'}")
        except Exception as e:
            logger.error(f"Error loading gender/age classifier: {e}")

    def _classify_person(self, frame: np.ndarray, bbox, track_id: int = None) -> str:
        """Clasifica persona por genero/edad usando DemographicsClassifier.

        Retorna categoria legacy ('Hombres', 'Mujeres', 'Ninos') para
        compatibilidad con el sistema existente de contadores.
        """
        if track_id is not None and hasattr(self, '_demographics'):
            result = self._demographics.classify(frame, bbox, track_id)
            # Log de clasificacion
            if hasattr(self, '_analytics_logger'):
                self._analytics_logger.log_demographic(
                    track_id, result['gender'], result['age_range']
                )
            return result.get('category', 'Hombres')
        return 'Hombres'

    async def send_alert(self, base64_img: str, text: str):
        """Envía una alerta con imagen"""
        payload = {
            "my-text": text, 
            "my-file": base64_img, 
            "type": "image/jpg",
            "compressed": "true",
            "quality": str(self.image_quality)
        }
        
        payload_size = len(base64_img) / 1024
        if self.debug_mode:
            print(f"📦 Tamaño del payload: {payload_size:.2f} KB")
        
        if payload_size > 500:
            logger.warning(f"Payload demasiado grande ({payload_size:.2f} KB)")
            payload = {
                "my-text": f"{text} (Imagen muy grande: {payload_size:.2f} KB)",
                "type": "text"
            }
        
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            try: 
                respuesta = await client.post(
                    "https://72.68.60.254:4000/bot/imgV2/number=120363402589311344@g.us",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                respuesta.raise_for_status()
                logger.info(f"✅ Alerta enviada: {respuesta.status_code}")
                return respuesta.json()
            except Exception as e:
                logger.error(f"❌ Error en envío: {e}")
                raise

    def send_alert_wrapper(self, base64_img: str, text: str, object_id: int):
        current_time = time.time()
        last_time = self.last_sent_time.get(object_id, 0)
        
        if current_time - last_time < self.send_cooldown:
            if self.debug_mode:
                print(f"⏳ Cooldown para objeto {object_id}")
            return
        
        def send_async():
            try:
                asyncio.run(self.send_alert(base64_img, text))
                self.last_sent_time[object_id] = current_time
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.send_alert(base64_img, text))
                        self.last_sent_time[object_id] = current_time
                    finally:
                        loop.close()
                else:
                    logger.error(f"Envío falló: {e}")
            except Exception as e:
                logger.error(f"Envío falló: {e}")
        
        thread = threading.Thread(target=send_async)
        thread.daemon = True
        thread.start()

    def create_annotated_image(self, frame: np.ndarray, staff_id: int, object_id: int, confidence: float = None) -> np.ndarray:
        annotated_frame = frame.copy()
        
        if object_id in self.active_tracks:
            track = self.active_tracks[object_id]
            x1, y1, x2, y2 = [int(v) for v in track['box']]
            
            # Color diferente por empleado (basado en ID)
            color_map = [
                (0, 255, 255),   # Amarillo - ID 0
                (255, 0, 0),     # Azul - ID 1
                (0, 255, 0),     # Verde - ID 2
                (255, 0, 255),   # Magenta - ID 3
                (0, 165, 255),   # Naranja - ID 4
                (255, 255, 0),   # Cyan - ID 5
                (128, 0, 128),   # Púrpura - ID 6
                (255, 192, 203)  # Rosa - ID 7
            ]
            
            color_idx = staff_id % len(color_map)
            box_color = color_map[color_idx]
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Etiqueta con nombre y confianza
            staff_name = self.get_staff_name(staff_id, confidence)
            label = f"{staff_name}"
            
            # Añadir tiempo en área si está dentro del ROI
            if 'entry_time' in track:
                current_time = time.time()
                time_in_roi = int(current_time - track['entry_time'])
                minutes = time_in_roi // 60
                seconds = time_in_roi % 60
                label += f" - {minutes}m {seconds}s"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Fondo para el texto
            cv2.rectangle(annotated_frame, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        (0, 0, 0), 
                        -1)
            
            # Texto
            cv2.putText(annotated_frame, label,
                      (x1, y1 - 5), font, font_scale,
                      (255, 255, 255), thickness)
        
        return annotated_frame

    def get_action_message(self, staff_id: int, event: str, time_in_roi: int = 0, confidence: float = None) -> str:
        staff_name = self.get_staff_display_name(staff_id)
        
        if event == 'entrada':
            return f"👤 {staff_name} entró en el área de oficina"
        elif event == 'salida':
            minutes = time_in_roi // 60
            if minutes == 1:
                return f"👤 {staff_name} salió después de {minutes} minuto"
            else:
                return f"👤 {staff_name} salió después de {minutes} minutos"
        elif event == 'alerta_periodica':
            minutes = time_in_roi // 60
            if minutes == 1:
                return f"⏰ {staff_name} lleva {minutes} minuto en el área"
            else:
                return f"⏰ {staff_name} lleva {minutes} minutos en el área"
        else:
            return f"{staff_name} - {event.upper()}"

    def save_staff_photo(self, frame: np.ndarray, staff_id: int, object_id: int, event: str, 
                         confidence: float = None, time_in_roi: int = 0):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            staff_name_safe = self.get_staff_display_name(staff_id).replace(' ', '_')
            
            event_dir = os.path.join("output/Personal", event.capitalize())
            os.makedirs(event_dir, exist_ok=True)
            
            filename = f"{staff_name_safe}_{event}_{object_id}_{timestamp}.jpg"
            filepath = os.path.join(event_dir, filename)
            
            annotated_frame = self.create_annotated_image(frame, staff_id, object_id, confidence)
            compressed_frame = self.compress_image(annotated_frame)
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success, buffer = cv2.imencode('.jpg', compressed_frame, encode_params)
            
            if success:
                imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                base64_size_kb = len(imagen_base64) / 1024
                
                if base64_size_kb > 500:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 40]
                    success, buffer = cv2.imencode('.jpg', compressed_frame, encode_params)
                    if success:
                        imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                        base64_size_kb = len(imagen_base64) / 1024
                
                message = self.get_action_message(staff_id, event, time_in_roi, confidence)
                self.send_alert_wrapper(imagen_base64, message, object_id)
                
                with open(filepath, 'wb') as f:
                    f.write(buffer)
                
                # Registrar en log
                self._log_staff_event(staff_id, event, time_in_roi, confidence)
                
                logger.info(f"✅ Foto de {event} guardada: {filename}")
                if self.debug_mode:
                    minutes = time_in_roi // 60
                    seconds = time_in_roi % 60
                    time_info = f" - Tiempo: {minutes}m {seconds}s" if time_in_roi > 0 else ""
                    print(f"📸 {message}{time_info} ({base64_size_kb:.2f} KB)")
                
                return True
            return False
        except Exception as e:
            logger.error(f"No se pudo guardar la foto: {e}")
            return False

    def _log_staff_event(self, staff_id: int, event: str, time_in_roi: int = 0, confidence: float = None):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        staff_name = self.get_staff_display_name(staff_id)
        minutes = time_in_roi // 60
        seconds = time_in_roi % 60
        
        log_entry = f"{ts},{self.frame_counter},{staff_id},{staff_name},{event},{minutes}:{seconds}"
        if confidence:
            log_entry += f",{confidence:.2f}"
        else:
            log_entry += ",N/A"
        
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(f"{log_entry}\n")

    def _get_cam_state(self, camera_id: Any) -> Dict[str, Any]:
        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = {
                'active_tracks': {},
                'next_id': 1,
                'track_history': defaultdict(lambda: deque(maxlen=30)),
                'movement_history': defaultdict(lambda: deque(maxlen=10)),
                'employee_counters': defaultdict(int),
                'counted_tracks': set(),
                'recent_counted_persons': deque(maxlen=30),
                'person_cooldown': defaultdict(int),
                'alert_minutes_sent': defaultdict(list),
                'sent_entry_photos': defaultdict(lambda: deque(maxlen=2)),
                'sent_exit_photos': defaultdict(lambda: deque(maxlen=2)),
                'last_sent_time': defaultdict(float),
                'last_processed_frame': None,
                'personas_en_area': 0,
                'entry_counts': {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0},
            }
        # Migrar estados que no tengan entry_counts
        state = self.camera_states[camera_id]
        if 'entry_counts' not in state:
            state['entry_counts'] = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}
        # Migrar analytics por camara
        if '_demographics' not in state:
            state['_demographics'] = DemographicsClassifier(
                gender_net=self._gender_net,
                age_net=self._age_net,
                face_cascade=self._face_cascade,
                face_detector_dnn=getattr(self, '_face_dnn', None),
                insightface_session=getattr(self, '_insightface_session', None),
                yunet=getattr(self, '_yunet', None),
                mivolo_session=getattr(self, '_mivolo_session', None),
                reidentifier=getattr(self, '_reidentifier', None),
            )
            state['_people_counter'] = PeopleCounter()
            state['_attendance_tracker'] = AttendanceTracker()
            state['_seller_efficiency'] = SellerEfficiency()
            state['_stock_monitor'] = StockMonitor()
            state['_analytics_logger'] = AnalyticsLogger(
                log_path=f"output/analytics_log_{camera_id}.jsonl"
            )
        return state

    def get_camera_processor(self, camera_id: Any):
        """Return or create a per-camera PersonAmazonas instance that shares the heavy model.

        This keeps tracking state isolated per camera while sharing the detection model.
        """
        if camera_id in self._camera_processors:
            return self._camera_processors[camera_id]

        # Crear nueva instancia ligera que comparte ambos modelos
        cam_proc = PersonAmazonas(
            client_id=f"{self.client_id}_{camera_id}",
            model_path=self.model_path,
            person_model_path=self.person_model_path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
            log_file=self.log_file,
            image_quality=self.image_quality,
            min_time_in_roi=self.min_time_in_roi,
            max_frames_out=self.max_frames_out,
            min_track_frames=self.min_track_frames,
            show_minimal_info=self.show_minimal_info,
            exit_frames_threshold=self.exit_frames_threshold,
            max_frames_without_detection=self.max_frames_without_detection,
            max_image_size=self.max_image_size,
            staff_names_file=None,
            shared_model=self.model,
            shared_person_model=self.person_model,
        )

        # Keep camera-specific state separate
        # Copy model-related metadata so camera instance can use class names and filters
        try:
            cam_proc.staff_names = dict(self.staff_names)
            cam_proc.all_classes = list(self.all_classes)
            cam_proc.roi_polygon = np.array(self.roi_polygon)
            # Los modulos de analytics ya se inicializan en __init__ con sus propias
            # instancias independientes, asi que cada camera_processor tiene su propio estado
        except Exception:
            pass

        self._camera_processors[camera_id] = cam_proc
        return cam_proc

    def _push_state(self, cam_state: Dict[str, Any]):
        # Guardar punteros actuales
        names = [
            'active_tracks', 'next_id', 'track_history', 'movement_history', 'employee_counters',
            'counted_tracks', 'recent_counted_persons', 'person_cooldown', 'alert_minutes_sent',
            'sent_entry_photos', 'sent_exit_photos', 'last_sent_time', 'last_processed_frame', 'personas_en_area',
            'entry_counts',
            '_demographics', '_people_counter', '_attendance_tracker',
            '_seller_efficiency', '_stock_monitor', '_analytics_logger',
        ]
        saved = {}
        for n in names:
            saved[n] = getattr(self, n, None)
            # asignar el estado de la cámara, si existe la clave
            setattr(self, n, cam_state.get(n))
        self._state_swap_stack.append(saved)

    def _pop_state(self):
        if not self._state_swap_stack:
            return
        saved = self._state_swap_stack.pop()
        for n, v in saved.items():
            setattr(self, n, v)

    def check_periodic_alerts(self, frame: np.ndarray):
        current_time = time.time()
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for track_id, track in list(self.active_tracks.items()):
            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            
            if is_inside and track['class'] in self.staff_names:
                if 'entry_time' not in track:
                    track['entry_time'] = current_time
                    track['last_alert_minute'] = 0
                
                time_in_roi = int(current_time - track['entry_time'])
                current_minute = time_in_roi // 60
                
                # Alertas a los 1, 3, 6, 9... minutos
                target_minutes = []
                minute = 1
                while minute <= current_minute:
                    target_minutes.append(minute)
                    minute += 3
                
                sent_minutes = self.alert_minutes_sent.get(track_id, [])
                
                for target_minute in target_minutes:
                    if target_minute not in sent_minutes:
                        success = self.save_staff_photo(
                            frame,
                            track['class'],
                            track_id,
                            'alerta_periodica',
                            track.get('confidence'),
                            time_in_roi
                        )
                        
                        if success:
                            if track_id not in self.alert_minutes_sent:
                                self.alert_minutes_sent[track_id] = []
                            
                            if target_minute not in self.alert_minutes_sent[track_id]:
                                self.alert_minutes_sent[track_id].append(target_minute)
                            
                            track['last_alert_minute'] = target_minute
                            
                            if self.debug_mode:
                                staff_name = self.get_staff_display_name(track['class'])
                                print(f"⏰ ALERTA: {staff_name} lleva {target_minute} minuto{'s' if target_minute > 1 else ''}")
            else:
                if 'entry_time' in track:
                    del track['entry_time']
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]

    def is_near_recent_counted(self, center: Tuple, threshold: int = 50) -> bool:
        for counted_id, counted_center, counted_frame in self.recent_counted_persons:
            distance = np.sqrt((center[0] - counted_center[0])**2 + (center[1] - counted_center[1])**2)
            if distance < threshold and (self.frame_counter - counted_frame) < 30:
                return True
        return False

    def validate_movement(self, track_id: int, current_pos: Tuple) -> bool:
        if track_id not in self.movement_history:
            self.movement_history[track_id].append(current_pos)
            return True
        
        positions = list(self.movement_history[track_id])
        if len(positions) < 3:
            self.movement_history[track_id].append(current_pos)
            return True
        
        first_pos = positions[0]
        distance = np.sqrt((current_pos[0] - first_pos[0])**2 + (current_pos[1] - first_pos[1])**2)
        
        self.movement_history[track_id].append(current_pos)
        return distance > 10

    def cleanup_undetected_tracks(self, current_detections: list):
        if not current_detections:
            if self.active_tracks and self.debug_mode:
                print(f"⚠️ No hay detecciones, limpiando tracks")
            self.active_tracks.clear()
            return
        
        track_ids = list(self.active_tracks.keys())
        unmatched_tracks = []
        
        for track_id in track_ids:
            track = self.active_tracks[track_id]
            track_box = track['box']
            
            has_match = False
            for det in current_detections:
                iou = self.calculate_iou(track_box, det['box'])
                if iou > 0.3:
                    has_match = True
                    break
            
            if not has_match:
                track['frames_without_detection'] = track.get('frames_without_detection', 0) + 1
                
                if track['frames_without_detection'] >= self.max_frames_without_detection:
                    unmatched_tracks.append(track_id)
            else:
                track['frames_without_detection'] = 0
        
        for track_id in unmatched_tracks:
            if self.debug_mode:
                track_info = self.active_tracks[track_id]
                staff_name = self.get_staff_display_name(track_info['class'])
                print(f"🗑️ {staff_name} eliminado - {self.max_frames_without_detection} frames sin detección")
            self._remove_track(track_id)

    def process_entry_exit_logic(self, frame: np.ndarray):
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        counted_in_frame = []
        tracks_to_remove = []

        self.person_count_inside = 0
        self.active_by_category = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}

        for track_id, track in list(self.active_tracks.items()):
            # Los cosmeticos (staff_id >= 0, modelo cosmeticos) no deben
            # seguir la logica de entrada/salida del ROI amarillo (que es
            # para PERSONAS). Saltarlos para que no se eliminen cuando
            # estan fuera del ROI amarillo.
            if track.get('class', -1) != -1:
                # Refrescar is_inside sin hacer contabilidad de entrada/salida
                track['is_inside'] = self.is_inside_polygon(
                    track['center'],
                    roi_polygon_points
                )
                continue

            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)

            previous_inside = track.get('is_inside', False)
            track['is_inside'] = is_inside

            if is_inside and not previous_inside:
                # ENTRADA del empleado
                track['entry_time'] = time.time()
                track['last_alert_minute'] = 0
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]

                if hasattr(self, 'last_processed_frame'):
                    self.save_staff_photo(
                        self.last_processed_frame,
                        track['class'],
                        track_id,
                        'entrada',
                        track.get('confidence')
                    )

                if self.debug_mode:
                    staff_name = self.get_staff_display_name(track['class'])
                    confidence = track.get('confidence', 0)
                    print(f"🚪 {staff_name} ({confidence:.2f}) entró al área")

                # Incrementar contador del empleado
                self.employee_counters[track['class']] += 1

                # Incrementar contador por categoría (usa la clasificación del track)
                category = track.get('category', 'Personas')
                self.entry_counts[category] = self.entry_counts.get(category, 0) + 1
            
            elif not is_inside and previous_inside:
                # SALIDA del empleado
                total_time = 0
                total_minutes = 0
                if 'entry_time' in track:
                    total_time = int(time.time() - track['entry_time'])
                    total_minutes = total_time // 60

                # Registrar visita en CSV solo para personas (class == -1).
                # Los cosmeticos no son "personas" y no deben aparecer en
                # el CSV de visitas.
                if track.get('class', -1) == -1 and total_time > 0:
                    gender = "Desconocido"
                    age_range = "Desconocido"
                    if hasattr(self, '_demographics'):
                        demo = self._demographics.get_cached(track_id)
                        if demo:
                            gender = demo.get('gender', 'Desconocido')
                            age_range = demo.get('age_range', 'Desconocido')
                    if hasattr(self, '_analytics_logger'):
                        try:
                            self._analytics_logger.log_visit_csv(
                                gender, age_range, total_time
                            )
                        except Exception as _e:
                            logger.error(f"Error CSV visita: {_e}")

                if 'entry_time' in track:
                    del track['entry_time']
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]

                if hasattr(self, 'last_processed_frame'):
                    self.save_staff_photo(
                        self.last_processed_frame,
                        track['class'],
                        track_id,
                        'salida',
                        track.get('confidence'),
                        total_time
                    )

                if self.debug_mode:
                    staff_name = self.get_staff_display_name(track['class'])
                    print(f"🚪 {staff_name} salió del área - Duró {total_minutes} minutos")
            
            if is_inside:
                track['has_been_inside'] = True
                track['frames_out_roi'] = 0
                track['frames_in_roi'] = track.get('frames_in_roi', 0) + 1
                self.person_count_inside += 1
                cat = track.get('category', 'Personas')
                self.active_by_category[cat] = self.active_by_category.get(cat, 0) + 1
            else:
                track['frames_out_roi'] = track.get('frames_out_roi', 0) + 1
                track['frames_in_roi'] = 0
                
                frames_in_roi = track.get('total_frames_in_roi', 0)
                frames_out_roi = track.get('frames_out_roi', 0)
                
                if (frames_in_roi >= self.min_time_in_roi and 
                    frames_out_roi >= 2 and
                    track['seen_frames'] >= self.min_track_frames and
                    not track.get('counted', False)):
                    
                    if not self.is_near_recent_counted(current_pos):
                        counted_in_frame.append(track_id)
                
                if track.get('has_been_inside', False) and track['frames_out_roi'] >= self.exit_frames_threshold:
                    tracks_to_remove.append(track_id)
                
                elif not track.get('has_been_inside', False) and track['frames_out_roi'] > self.max_frames_out:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)
        
        for track_id in counted_in_frame:
            if self._count_staff_safe(track_id):
                if track_id in self.active_tracks:
                    self._remove_track(track_id)
        
        return self.person_count_inside

    def _count_staff_safe(self, track_id: int) -> bool:
        if track_id not in self.active_tracks:
            return False
        
        track = self.active_tracks[track_id]
        
        if track.get('counted', False) or track_id in self.counted_tracks:
            return False
        
        if not self.validate_movement(track_id, track['center']):
            if self.debug_mode:
                print(f"⚠️ Movimiento no válido - ignorando")
            return False
        
        self.counted_tracks.add(track_id)
        track['counted'] = True
        track['counted_at_frame'] = self.frame_counter
        
        self.recent_counted_persons.append((track_id, track['center'], self.frame_counter))
        
        self.personas_en_area += 1
        self.last_counted_frame = self.frame_counter
        self.last_counted_id = track_id
        
        staff_name = self.get_staff_display_name(track['class'])
        confidence = track.get('confidence', 0)
        
        print(f"\n{'='*60}")
        print(f"🎉 {staff_name} ({confidence:.2f}) EN ÁREA!")
        print(f"   Total personal en área: {self.personas_en_area}")
        print(f"   Contador de {staff_name}: {self.employee_counters[track['class']]}")
        print(f"{'='*60}\n")
        
        return True

    def _register_product_track(self, track_id: int, class_id: int) -> bool:
        """Registra un track_id de producto cosmetico como unico.

        Retorna True la PRIMERA vez que ve este track_id, False en repeticiones.
        Mantiene tambien un contador por SKU (nombre legible del producto).
        """
        if track_id in self._product_track_ids:
            return False
        self._product_track_ids.add(track_id)
        sku = self.staff_names.get(class_id, f"Clase_{class_id}")
        self._product_counts_by_sku[sku] = (
            self._product_counts_by_sku.get(sku, 0) + 1
        )
        return True

    def _count_active_products(self) -> int:
        """Cuenta cuantos productos (class != -1) estan activos en este frame."""
        return sum(
            1 for t in self.active_tracks.values()
            if t.get('class', -1) != -1
        )

    def _detect_pickup_events(self, image: np.ndarray):
        """Detecta cuando una persona TOMA o MUESTRA un producto cosmetico.

        Tres criterios (cualquiera dispara el evento):
          A) SKU solapado con persona (IoU>0.10) y SKU<40% del area persona.
             -> Persona sosteniendo el producto contra el cuerpo.
          B) SKU fuera de TODOS los ROIs y hay >=1 persona activa.
             -> Persona movio el producto fuera del estante (lo agarro).
          C) SKU detectado cerca de la mano: distancia centro_sku ↔
             centro_persona < diagonal_persona * 0.8 y NO esta en ROI verde
             (estante).
             -> Producto mostrado frente a camara con brazo extendido.

        Cada par (persona, sku) emite UNA alerta + foto con cooldown de 30s.
        """
        if image is None or image.size == 0:
            return

        # Separar tracks activos: personas vs cosmeticos
        person_tracks = [
            (tid, t) for tid, t in self.active_tracks.items()
            if t.get('class', -1) == -1
        ]
        sku_tracks = [
            (tid, t) for tid, t in self.active_tracks.items()
            if t.get('class', -1) != -1
        ]
        if not sku_tracks:
            return

        # Polygon del estante (ROI verde) para distinguir "en estante"
        # vs "siendo mostrado".
        esc_poly = None
        if hasattr(self, 'roi_escaparate') and self.roi_escaparate is not None:
            esc_poly = self.roi_escaparate.reshape((-1, 1, 2))

        now = time.time()
        cooldown_sec = 30.0

        for sku_tid, sku_track in sku_tracks:
            sku_box = sku_track.get('box')
            if sku_box is None:
                continue
            sku_area = max(1.0, (sku_box[2] - sku_box[0]) *
                           (sku_box[3] - sku_box[1]))
            sku_cx = (sku_box[0] + sku_box[2]) / 2.0
            sku_cy = (sku_box[1] + sku_box[3]) / 2.0

            # ¿El SKU esta dentro del estante (ROI verde)?
            sku_in_shelf = False
            if esc_poly is not None:
                sku_in_shelf = (
                    cv2.pointPolygonTest(
                        esc_poly, (int(sku_cx), int(sku_cy)), False
                    ) >= 0
                )

            best_person_tid = None
            best_score = 0.0
            best_person_box = None
            trigger_reason = ""

            # ── Criterio A: solape (sostenido contra el cuerpo) ──
            for p_tid, p_track in person_tracks:
                p_box = p_track.get('box')
                if p_box is None:
                    continue
                iou = self.calculate_iou(sku_box, p_box)
                if iou < 0.10:
                    continue
                p_area = max(1.0, (p_box[2] - p_box[0]) *
                             (p_box[3] - p_box[1]))
                if (sku_area / p_area) >= 0.40:
                    continue
                if iou > best_score:
                    best_score = iou
                    best_person_tid = p_tid
                    best_person_box = p_box
                    trigger_reason = "sostenido"

            # ── Criterio B/C: SKU fuera del estante + persona cercana ──
            if best_person_tid is None and not sku_in_shelf:
                # Buscar la persona MAS CERCANA al SKU
                closest_dist = float('inf')
                for p_tid, p_track in person_tracks:
                    p_box = p_track.get('box')
                    if p_box is None:
                        continue
                    p_cx = (p_box[0] + p_box[2]) / 2.0
                    p_cy = (p_box[1] + p_box[3]) / 2.0
                    p_w = p_box[2] - p_box[0]
                    p_h = p_box[3] - p_box[1]
                    p_diag = (p_w ** 2 + p_h ** 2) ** 0.5
                    dist = ((sku_cx - p_cx) ** 2 +
                            (sku_cy - p_cy) ** 2) ** 0.5
                    # Considerar "cerca" si distancia <= 1.2x la diagonal
                    # de la persona (cubre brazo extendido).
                    if dist > p_diag * 1.2:
                        continue
                    if dist < closest_dist:
                        closest_dist = dist
                        best_person_tid = p_tid
                        best_person_box = p_box
                        trigger_reason = "mostrado"

            # ── Caso fallback: SKU mostrado a camara sin persona detectada ──
            # (la persona puede estar fuera del frame con el brazo metido)
            if (best_person_tid is None and not sku_in_shelf
                    and sku_area > (image.shape[0] * image.shape[1] * 0.02)):
                # SKU ocupa >2% del frame Y NO esta en estante => alguien
                # lo movio. Asignar persona "anonima" tid=0.
                best_person_tid = 0
                best_person_box = sku_box
                trigger_reason = "frente_camara"

            if best_person_tid is None:
                continue

            # Cooldown: no emitir el mismo par persona-sku con demasiada
            # frecuencia. Permite que la misma persona tome el mismo SKU
            # otra vez si pasaron >30s.
            pair_key = (best_person_tid, sku_tid)
            last = self._pickup_last_emit.get(pair_key, 0.0)
            if (now - last) < cooldown_sec:
                continue

            # Emitir alerta
            self._pickup_last_emit[pair_key] = now
            self._pickup_emitted.add(pair_key)

            staff_id = sku_track.get('class', -1)
            sku_name = self.staff_names.get(staff_id, f"Clase_{staff_id}")
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Crop: union persona + SKU con margen para que se vea contexto
            try:
                ux1 = int(min(sku_box[0], best_person_box[0]))
                uy1 = int(min(sku_box[1], best_person_box[1]))
                ux2 = int(max(sku_box[2], best_person_box[2]))
                uy2 = int(max(sku_box[3], best_person_box[3]))
                # Margen 10%
                hh, ww = image.shape[:2]
                mx = int((ux2 - ux1) * 0.10)
                my = int((uy2 - uy1) * 0.10)
                ux1 = max(0, ux1 - mx)
                uy1 = max(0, uy1 - my)
                ux2 = min(ww, ux2 + mx)
                uy2 = min(hh, uy2 + my)
                crop = image[uy1:uy2, ux1:ux2]
                if crop.size == 0:
                    crop = image
            except Exception:
                crop = image

            # Guardar a disco para historial
            screenshot_path = ""
            try:
                fname = f"pickup_{int(now)}_{best_person_tid}_{sku_tid}.jpg"
                screenshot_path = os.path.join(self._pickup_dir, fname)
                cv2.imwrite(screenshot_path, crop)
            except Exception:
                screenshot_path = ""

            # Codificar JPEG -> base64 para enviar al cliente
            img_b64 = ""
            try:
                ok, buf = cv2.imencode(".jpg", crop,
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            except Exception:
                img_b64 = ""

            desc_map = {
                "sostenido": f"Persona tomó producto: {sku_name}",
                "mostrado": f"Persona muestra producto: {sku_name}",
                "frente_camara": f"Producto frente a cámara: {sku_name}",
            }
            description = desc_map.get(
                trigger_reason, f"Producto detectado: {sku_name}"
            )

            self._pending_pickup_alerts.append({
                "event_type": "toma de orden",
                "class_name": sku_name,
                "description": description,
                "timestamp": timestamp_str,
                "image_base64": img_b64,
                "crop_image": img_b64,
                "screenshot_path": screenshot_path,
                "person_track_id": int(best_person_tid),
                "sku_track_id": int(sku_tid),
                "trigger": trigger_reason,
            })

            if self.debug_mode:
                print(f"📸 PICKUP [{trigger_reason}]: persona#{best_person_tid} "
                      f"con {sku_name} (sku#{sku_tid})")

    def _remove_track(self, track_id: int):
        if track_id in self.active_tracks:
            track = self.active_tracks[track_id]
            staff_id = track['class']
            staff_name = self.get_staff_display_name(staff_id)

            # Si es persona (class == -1) y tiene entry_time, registrar
            # visita en CSV aunque no haya habido SALIDA formal (el track
            # se puede limpiar por timeout antes de cruzar el ROI).
            if staff_id == -1 and 'entry_time' in track:
                try:
                    total_time = int(time.time() - track['entry_time'])
                    if total_time > 0:
                        gender = "Desconocido"
                        age_range = "Desconocido"
                        if hasattr(self, '_demographics'):
                            demo = self._demographics.get_cached(track_id)
                            if demo:
                                gender = demo.get('gender', 'Desconocido')
                                age_range = demo.get('age_range',
                                                     'Desconocido')
                        if hasattr(self, '_analytics_logger'):
                            self._analytics_logger.log_visit_csv(
                                gender, age_range, total_time
                            )
                except Exception as _e:
                    logger.error(f"Error CSV visita en remove: {_e}")

            if self.debug_mode:
                print(f"✅ {staff_name} eliminado del seguimiento")
            del self.active_tracks[track_id]
        
        if track_id in self.track_history:
            del self.track_history[track_id]
        
        if track_id in self.movement_history:
            del self.movement_history[track_id]
        
        if track_id in self.person_cooldown:
            del self.person_cooldown[track_id]
        
        if track_id in self.alert_minutes_sent:
            del self.alert_minutes_sent[track_id]
        # clean up class history
        if track_id in self.track_class_history:
            try:
                del self.track_class_history[track_id]
            except Exception:
                pass

        # Limpiar analytics
        if hasattr(self, '_attendance_tracker'):
            self._attendance_tracker.remove_track(track_id)

    def cleanup_stale_tracks(self):
        current_frame = self.frame_counter
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            frames_since_last = current_frame - track['last_seen']
            
            if (frames_since_last > 30 or 
                (track.get('frames_out_roi', 0) > 50 and not track.get('has_been_inside', False))):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if self.debug_mode:
                staff_id = self.active_tracks[track_id]['class'] if track_id in self.active_tracks else None
                staff_name = self.get_staff_display_name(staff_id) if staff_id else "desconocido"
                print(f"🗑️ {staff_name} eliminado (inactivo)")
            self._remove_track(track_id)

    def match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.active_tracks or not detections:
            return [], list(range(len(detections))), list(self.active_tracks.keys())
        
        similarity_matrix = np.zeros((len(self.active_tracks), len(detections)))
        track_ids = list(self.active_tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track_box = self.active_tracks[track_id]['box']
            for j, det in enumerate(detections):
                similarity_matrix[i, j] = self.calculate_iou(track_box, det['box'])
        
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        iou_threshold = 0.3
        
        while True:
            if similarity_matrix.size == 0:
                break
                
            max_iou = np.max(similarity_matrix)
            if max_iou < iou_threshold:
                break
            
            i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            
            matched_pairs.append((track_ids[i], j))
            unmatched_tracks.remove(i)
            unmatched_detections.remove(j)
            
            similarity_matrix[i, :] = 0
            similarity_matrix[:, j] = 0
        
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_track_ids

    def update_tracks(self, detections: list):
        self.cleanup_undetected_tracks(detections)
        self.cleanup_stale_tracks()
        
        filtered_detections = []
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for det in detections:
            center = det['center']
            distance_to_roi = cv2.pointPolygonTest(roi_polygon_points, (int(center[0]), int(center[1])), True)
            
            if distance_to_roi > -50:
                filtered_detections.append(det)
        
        current_detections = []
        for det in filtered_detections:
            current_detections.append({
                'box': det['box'],
                'class': det['class'],
                'center': det['center'],
                'confidence': det.get('confidence', 0.5)
            })
        
        if self.active_tracks and current_detections:
            matched_pairs, unmatched_detections, unmatched_tracks = self.match_detections_to_tracks(current_detections)
            
            for track_id, det_idx in matched_pairs:
                det = current_detections[det_idx]
                self.active_tracks[track_id].update({
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': self.frame_counter,
                    'seen_frames': self.active_tracks[track_id]['seen_frames'] + 1,
                    'confidence': max(self.active_tracks[track_id].get('confidence', 0), det['confidence'])
                })
                # Re-clasificar mientras la categoria no sea definitiva.
                # Seguimos clasificando si esta en 'Personas' o 'Desconocido'
                # (estado pendiente). Solo dejamos de re-clasificar cuando
                # el accumulator commitea Hombres/Mujeres/Ninos.
                _cur_cat = self.active_tracks[track_id].get('category', 'Personas')
                if (_cur_cat in ('Personas', 'Desconocido')
                        and self.frame_counter % self._classify_every_n == 0
                        and hasattr(self, 'last_processed_frame')
                        and self.last_processed_frame is not None):
                    new_cat = self._classify_person(self.last_processed_frame, det['box'], track_id)
                    # Solo guardamos categorias firmes; 'Desconocido' es estado
                    # transitorio del classifier mientras acumula muestras.
                    if new_cat in ('Hombres', 'Mujeres', 'Ninos'):
                        self.active_tracks[track_id]['category'] = new_cat
                # Append class observation for stability voting
                self.track_history[track_id].append(det['center'])
                try:
                    self.track_class_history[track_id].append(det['class'])
                except Exception:
                    pass
                # Update stable class if threshold reached
                try:
                    self._update_track_class_stability(track_id)
                except Exception:
                    pass
            
            for track_id in unmatched_tracks:
                if track_id in self.active_tracks:
                    self.active_tracks[track_id]['last_seen'] = self.frame_counter
            
            for det_idx in unmatched_detections:
                det = current_detections[det_idx]
                self._create_new_track(det)
        
        elif current_detections:
            for det in current_detections:
                self._create_new_track(det)

    def _create_new_track(self, detection: Dict):
        new_id = self.next_id
        self.next_id += 1

        center = detection['center']
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        is_inside = self.is_inside_polygon(center, roi_polygon_points)

        # Clasificar persona (género/edad). Si el classifier devuelve
        # "Desconocido" (no convergio aun), dejamos category='Personas'
        # como estado pendiente para que se siga reclasificando en frames
        # siguientes (en _update_tracks_botsort).
        category = 'Personas'
        if hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
            cls_result = self._classify_person(
                self.last_processed_frame, detection['box'], new_id
            )
            if cls_result in ('Hombres', 'Mujeres', 'Ninos'):
                category = cls_result

        # Registrar en contador segun tipo de track:
        #  - class == -1  -> persona (YOLO COCO) => contador de personas
        #  - class >= 0   -> producto cosmetico (modelo amazonas/cosmeticos)
        _tclass = detection.get('class', -1)
        if _tclass == -1:
            if hasattr(self, '_people_counter'):
                is_new = self._people_counter.register_track(new_id)
                if is_new and hasattr(self, '_analytics_logger'):
                    demo = self._demographics.get_cached(new_id) if hasattr(self, '_demographics') else None
                    self._analytics_logger.log_person_detected(
                        new_id,
                        gender=demo.get('gender', '') if demo else '',
                        age_range=demo.get('age_range', '') if demo else '',
                    )
        else:
            # Producto: registrar track unico y contar por SKU
            self._register_product_track(new_id, _tclass)

        track_data = {
            'class': detection['class'],
            'box': detection['box'],
            'center': center,
            'last_seen': self.frame_counter,
            'seen_frames': 1,
            'counted': False,
            'is_inside': is_inside,
            'has_been_inside': is_inside,
            'frames_in_roi': 1 if is_inside else 0,
            'total_frames_in_roi': 0,
            'frames_out_roi': 0 if is_inside else 1,
            'entry_frame': self.frame_counter if is_inside else None,
            'frames_without_detection': 0,
            'confidence': detection.get('confidence', 0.5),
            'category': category,
        }
        
        if is_inside:
            track_data['entry_time'] = time.time()
            track_data['last_alert_minute'] = 0
        
        self.active_tracks[new_id] = track_data
        self.track_history[new_id].append(center)
        # Initialize class history for stabilization
        try:
            self.track_class_history[new_id].append(detection['class'])
            self._update_track_class_stability(new_id)
        except Exception:
            pass
        
        if self.debug_mode and is_inside:
            staff_name = self.get_staff_display_name(detection['class'])
            confidence = detection.get('confidence', 0)
            print(f"🆕 {staff_name} ({confidence:.2f}) detectado en ROI")

    def _update_tracks_botsort(self, detections: list):
        """Actualiza tracks usando los IDs estables de BoTSORT.

        A diferencia del matching manual por IoU, BoTSORT ya asigna IDs
        consistentes usando Kalman filter + IoU + apariencia. Solo necesitamos
        sincronizar nuestro dict active_tracks con esos IDs.
        """
        current_botsort_ids = set()

        for det in detections:
            tid = det.get('track_id')
            if tid is None:
                # Sin track ID (frame sin tracking), usar fallback
                self._create_new_track(det)
                continue

            current_botsort_ids.add(tid)

            if tid in self.active_tracks:
                # Track existente: actualizar posición y metadata
                track = self.active_tracks[tid]
                track.update({
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': self.frame_counter,
                    'seen_frames': track['seen_frames'] + 1,
                    'confidence': det['confidence'],
                    'frames_without_detection': 0,
                })
                self.track_history[tid].append(det['center'])

                # Re-clasificar mientras la categoria no sea definitiva.
                _cur_cat = track.get('category', 'Personas')
                if (_cur_cat in ('Personas', 'Desconocido')
                        and self.frame_counter % self._classify_every_n == 0
                        and self.last_processed_frame is not None):
                    new_cat = self._classify_person(self.last_processed_frame, det['box'], tid)
                    if new_cat in ('Hombres', 'Mujeres', 'Ninos'):
                        track['category'] = new_cat

                # Registrar track segun tipo (persona o producto cosmetico)
                _tclass = track.get('class', -1)
                if _tclass == -1:
                    if hasattr(self, '_people_counter'):
                        self._people_counter.register_track(tid)
                else:
                    self._register_product_track(tid, _tclass)
            else:
                # Nuevo track de BoTSORT: crear entrada
                center = det['center']
                roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
                is_inside = self.is_inside_polygon(center, roi_polygon_points)

                category = 'Personas'
                if self.last_processed_frame is not None:
                    cls_result = self._classify_person(
                        self.last_processed_frame, det['box'], tid
                    )
                    if cls_result in ('Hombres', 'Mujeres', 'Ninos'):
                        category = cls_result

                self.active_tracks[tid] = {
                    'class': det['class'],
                    'box': det['box'],
                    'center': center,
                    'last_seen': self.frame_counter,
                    'seen_frames': 1,
                    'counted': False,
                    'is_inside': is_inside,
                    'has_been_inside': is_inside,
                    'frames_in_roi': 1 if is_inside else 0,
                    'total_frames_in_roi': 0,
                    'frames_out_roi': 0 if is_inside else 1,
                    'entry_frame': self.frame_counter if is_inside else None,
                    'frames_without_detection': 0,
                    'confidence': det.get('confidence', 0.5),
                    'category': category,
                }
                if is_inside:
                    self.active_tracks[tid]['entry_time'] = time.time()
                    self.active_tracks[tid]['last_alert_minute'] = 0

                self.track_history[tid].append(center)

                # Registrar track segun tipo (persona o producto cosmetico)
                _tclass = det.get('class', -1)
                if _tclass == -1:
                    if hasattr(self, '_people_counter'):
                        is_new = self._people_counter.register_track(tid)
                        if is_new and hasattr(self, '_analytics_logger'):
                            demo = self._demographics.get_cached(tid) if hasattr(self, '_demographics') else None
                            self._analytics_logger.log_person_detected(
                                tid,
                                gender=demo.get('gender', '') if demo else '',
                                age_range=demo.get('age_range', '') if demo else '',
                            )
                else:
                    self._register_product_track(tid, _tclass)

                if self.debug_mode and is_inside:
                    print(f"🆕 [{category}] track_id={tid} ({det.get('confidence', 0):.2f}) en ROI")

        # Limpiar tracks que BoTSORT ya no reporta.
        # Cosmeticos (class != -1) tienen el doble de gracia que personas
        # porque pueden ser ocluidos transitoriamente por la mano del cliente
        # mientras agarran el producto. Los queremos mantener varios frames
        # mas antes de eliminarlos.
        stale_ids = []
        for tid in list(self.active_tracks.keys()):
            if tid not in current_botsort_ids:
                track = self.active_tracks[tid]
                track['frames_without_detection'] = track.get('frames_without_detection', 0) + 1
                # Personas: limite estandar; cosmeticos: 6x mas tolerante
                # (oclusion intermitente cuando la mano gira el producto).
                if track.get('class', -1) == -1:
                    limit = self.max_frames_without_detection
                else:
                    limit = self.max_frames_without_detection * 6
                if track['frames_without_detection'] >= limit:
                    stale_ids.append(tid)

        for tid in stale_ids:
            self._remove_track(tid)

    def draw_detections(self, image: np.ndarray, persons_inside: int) -> np.ndarray:
        # Dibujar ROI amarillo (personas/objetos)
        roi_overlay = image.copy()
        cv2.fillPoly(roi_overlay, [self.roi_polygon], (0, 255, 255, 100))
        cv2.addWeighted(roi_overlay, 0.3, image, 0.7, 0, image)
        cv2.polylines(image, [self.roi_polygon], isClosed=True, color=(0, 255, 255), thickness=3)

        for x, y in self.roi_polygon:
            cv2.circle(image, (x, y), 8, (255, 0, 0), -1)
            cv2.circle(image, (x, y), 8, (255, 255, 255), 2)

        # Dibujar ROI verde (escaparate/estante)
        if hasattr(self, 'roi_escaparate') and self.roi_escaparate is not None:
            esc_overlay = image.copy()
            cv2.fillPoly(esc_overlay, [self.roi_escaparate], (0, 255, 0, 100))
            cv2.addWeighted(esc_overlay, 0.2, image, 0.8, 0, image)
            cv2.polylines(image, [self.roi_escaparate], isClosed=True, color=(0, 255, 0), thickness=3)
            for x, y in self.roi_escaparate:
                cv2.circle(image, (x, y), 8, (0, 200, 0), -1)
                cv2.circle(image, (x, y), 8, (255, 255, 255), 2)

        # Dibujar ROI violeta 1 (area de prueba de productos 1)
        if hasattr(self, 'roi_prueba1') and self.roi_prueba1 is not None:
            p1_overlay = image.copy()
            cv2.fillPoly(p1_overlay, [self.roi_prueba1], (211, 0, 148))
            cv2.addWeighted(p1_overlay, 0.2, image, 0.8, 0, image)
            cv2.polylines(image, [self.roi_prueba1], isClosed=True, color=(211, 0, 148), thickness=3)
            for x, y in self.roi_prueba1:
                cv2.circle(image, (x, y), 8, (211, 0, 148), -1)
                cv2.circle(image, (x, y), 8, (255, 255, 255), 2)

        # Dibujar ROI violeta 2 (area de prueba de productos 2)
        if hasattr(self, 'roi_prueba2') and self.roi_prueba2 is not None:
            p2_overlay = image.copy()
            cv2.fillPoly(p2_overlay, [self.roi_prueba2], (211, 0, 148))
            cv2.addWeighted(p2_overlay, 0.2, image, 0.8, 0, image)
            cv2.polylines(image, [self.roi_prueba2], isClosed=True, color=(211, 0, 148), thickness=3)
            for x, y in self.roi_prueba2:
                cv2.circle(image, (x, y), 8, (211, 0, 148), -1)
                cv2.circle(image, (x, y), 8, (255, 255, 255), 2)

        # Colores por categoría (personas)
        category_colors = {
            'Hombres': (255, 180, 0),    # Azul claro
            'Mujeres': (180, 0, 255),    # Rosa/Magenta
            'Niños':   (0, 255, 180),    # Verde/Cyan
            'Personas': (0, 255, 255),   # Amarillo
        }

        # IDs de clases detectadas.
        # Antes: {2,3} = CLIENTE_TOMA_PRODUCTO_*, 4 = ESTANTE (modelo 'amazonas').
        # El nuevo modelo 'cosmeticos' no tiene esas clases (2/3/4 son
        # productos reales: aceites). Se dejan vacios para que ningun SKU
        # se dibuje como 'interaccion' o 'estante'.
        _INTERACTION_CLASS_IDS: set = set()
        _ESTANTE_CLASS_ID = -99  # nunca coincidente

        # Preparar polígono ROI para filtrar dibujo
        roi_pts = self.roi_polygon.reshape((-1, 1, 2)) if self.roi_polygon is not None else None

        # Dibujar en 2 pasadas para que los SKUs cosmeticos queden ENCIMA
        # del bbox de la persona cuando los tiene en la mano:
        #   Pasada 1: personas (class == -1)
        #   Pasada 2: cosmeticos (class >= 0) -> dibujados al final
        items_all = [
            (tid, obj) for tid, obj in self.active_tracks.items()
            if not obj.get('counted', False)
        ]
        # Lista de bboxes de personas para detectar overlap
        person_boxes_for_draw = [
            obj['box'] for _, obj in items_all
            if obj.get('class', -1) == -1 and 'box' in obj
        ]
        items_persons = [
            it for it in items_all if it[1].get('class', -1) == -1
        ]
        items_skus = [
            it for it in items_all if it[1].get('class', -1) != -1
        ]
        ordered_items = items_persons + items_skus

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        for _tid, obj in ordered_items:
            staff_id_pre = obj.get('class', -1)
            # Filtrar por ROI amarillo SOLO a personas (-1).
            if staff_id_pre == -1 and roi_pts is not None:
                cx, cy = obj.get('center', (0, 0))
                if cv2.pointPolygonTest(roi_pts, (int(cx), int(cy)), False) < 0:
                    continue

            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            staff_id = obj.get('class', -1)
            category = obj.get('category', 'Personas')
            confidence = obj.get('confidence', 0)

            demo = None
            if staff_id == -1 and hasattr(self, '_demographics'):
                demo = self._demographics.get_cached(_tid)

            def _demo_suffix(d):
                if not d:
                    return ""
                g = d.get('gender', 'Desconocido')
                a = d.get('age_range', 'Desconocido')
                if g == 'Desconocido' and a == 'Desconocido':
                    return " [?]"
                return f" [{g[:1]}|{a}]"

            is_sku = staff_id != -1 and staff_id not in _INTERACTION_CLASS_IDS \
                and staff_id != _ESTANTE_CLASS_ID

            if staff_id == -1:
                color = category_colors.get(category, (0, 255, 0))
                text = f"{category} {confidence:.2f}{_demo_suffix(demo)}"
                thickness = 2
            elif staff_id in _INTERACTION_CLASS_IDS:
                color = (0, 0, 255)
                label = self.staff_names.get(staff_id, f'Clase_{staff_id}')
                text = f"{label} {confidence:.2f}{_demo_suffix(demo)}"
                thickness = 3
            elif staff_id == _ESTANTE_CLASS_ID:
                color = (200, 200, 200)
                text = f"Estante {confidence:.2f}"
                thickness = 2
            else:
                # SKU cosmetico: bbox grueso y brillante para que se vea
                # claramente sobre el bbox de la persona.
                color = (0, 165, 255)  # Naranja vibrante BGR
                label = self.staff_names.get(staff_id, f'Clase_{staff_id}')
                text = f"{label} {confidence:.2f}"
                thickness = 4

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            if 'entry_time' in obj:
                current_time = time.time()
                time_in_roi = int(current_time - obj['entry_time'])
                minutes = time_in_roi // 60
                seconds = time_in_roi % 60
                text += f" - {minutes}m {seconds}s"

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )

            # Posicion del texto: por defecto arriba del bbox.
            # Para SKUs cuyo top esta dentro del bbox de una persona,
            # poner el texto DEBAJO del bbox para que no quede tapado.
            put_below = False
            if is_sku:
                for pb in person_boxes_for_draw:
                    if (pb[0] <= x1 <= pb[2] and pb[1] <= y1 <= pb[3]):
                        put_below = True
                        break

            if put_below:
                ty1 = min(image.shape[0] - text_height - 4, y2 + 2)
                ty2 = ty1 + text_height + 10
                tx1 = x1
                tx2 = x1 + text_width
                cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
                cv2.putText(
                    image, text, (tx1, ty1 + text_height + 4),
                    font, font_scale, (0, 200, 255) if is_sku else (255, 255, 255),
                    font_thickness
                )
            else:
                cv2.rectangle(
                    image, (x1, y1 - text_height - 10),
                    (x1 + text_width, y1), (0, 0, 0), -1
                )
                cv2.putText(
                    image, text, (x1, y1 - 5),
                    font, font_scale,
                    (0, 200, 255) if is_sku else (255, 255, 255),
                    font_thickness
                )
        
        # ── Overlay de analitica retail ──
        if self.show_minimal_info:
            # 1. Panel demografico (izquierda)
            if hasattr(self, '_demographics'):
                demo_counts = self._demographics.get_counts()
                image = draw_demographics_panel(image, demo_counts, x=10, y=10)

            # 2. Personas totales + atendidas (superior central).
            # Contar solo personas activas (class == -1), excluyendo
            # productos cosmeticos. Si no hay active_tracks usamos el
            # contador heredado persons_inside como fallback.
            total_unique = 0
            attended = 0
            active_persons = sum(
                1 for t in self.active_tracks.values()
                if t.get('class', -1) == -1 and t.get('is_inside', False)
            )
            active_now = active_persons if self.active_tracks else persons_inside
            if hasattr(self, '_people_counter'):
                total_unique = self._people_counter.total_unique
            if hasattr(self, '_attendance_tracker'):
                attended = self._attendance_tracker.get_stats().get('attended_count', 0)
            image = draw_people_total(image, total_unique, attended, active_now)

            # 2b. Productos totales (cosmeticos) - panel debajo de Personas
            products_total = len(getattr(self, '_product_track_ids', set()))
            products_active = self._count_active_products()
            image = draw_products_total(
                image, products_total, products_active, by_sku=None, y=55
            )

            # 3. Dashboard de vendedores (derecha)
            if hasattr(self, '_seller_efficiency'):
                dashboard = self._seller_efficiency.get_dashboard(max(total_unique, 1))
                if dashboard:
                    image = draw_seller_dashboard(image, dashboard)

            # 4. Lineas de proximidad vendedor-cliente
            if hasattr(self, '_attendance_tracker'):
                pairs = self._attendance_tracker.get_active_proximity_pairs()
                if pairs:
                    image = draw_proximity_lines(image, pairs, self.active_tracks)

            # 5. Indicadores de stock en ROIs de producto
            if hasattr(self, '_stock_monitor'):
                stock_rois = self._stock_monitor.get_rois()
                if stock_rois:
                    image = draw_stock_indicators(image, stock_rois)

            # 6. Banner de premio (si esta activo)
            if hasattr(self, '_seller_efficiency'):
                award = self._seller_efficiency.get_active_award()
                if award:
                    image = draw_award_banner(image, award)

        return image

    def process_frame(self, image: np.ndarray, roi=None, activate_roi=False, camera_id: Any = 1, roi_escaparate=None, roi_prueba1=None, roi_prueba2=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Solo el modelo de personas es estrictamente necesario.
        # El de cosmeticos puede estar None si cosmetics_enabled=False.
        if self.person_model is None:
            raise RuntimeError("Modelo de personas no inicializado")

        if roi is not None and isinstance(roi, list) and len(roi) >= 3:
            self.roi_polygon = np.array(roi, np.int32)
            if self.debug_mode:
                print(f"📍 ROI actualizado: {len(roi)} puntos")

        # ROI verde para escaparate/estante
        if roi_escaparate is not None and isinstance(roi_escaparate, list) and len(roi_escaparate) >= 3:
            self.roi_escaparate = np.array(roi_escaparate, np.int32)
        else:
            # Cliente desactivo el ROI -> limpiar siempre, no solo la 1a vez.
            # Antes usaba 'elif not hasattr' y el poligono quedaba cacheado,
            # por lo que al apagar en el cliente el ROI seguia pintandose.
            self.roi_escaparate = None

        # ROIs violeta para areas de prueba de productos
        if roi_prueba1 is not None and isinstance(roi_prueba1, list) and len(roi_prueba1) >= 3:
            self.roi_prueba1 = np.array(roi_prueba1, np.int32)
        else:
            self.roi_prueba1 = None

        if roi_prueba2 is not None and isinstance(roi_prueba2, list) and len(roi_prueba2) >= 3:
            self.roi_prueba2 = np.array(roi_prueba2, np.int32)
        else:
            self.roi_prueba2 = None

        # Usar estado por cámara para mantener separación de tracks
        cam_state = self._get_cam_state(camera_id)

        # Evitar solapamiento concurrente entre llamadas a process_frame
        with self._process_lock:
            self._push_state(cam_state)

            try:
                self.frame_counter += 1
                self.last_processed_frame = image.copy()

                # Detectar sobre la imagen COMPLETA (sin máscara) para evitar
                # artefactos en los bordes del ROI que generan falsos positivos.
                inference_image = image

                # Preparar polígono ROI para filtrado espacial
                roi_active = activate_roi and hasattr(self, 'roi_polygon') and self.roi_polygon is not None
                roi_poly = self.roi_polygon.reshape((-1, 1, 2)) if roi_active else None

                # Preparar polígonos de TODOS los ROIs (verde, amarillo, violetas).
                # Un cosmetico detectado dentro de CUALQUIERA de ellos es valido.
                esc_poly = None
                if hasattr(self, 'roi_escaparate') and self.roi_escaparate is not None:
                    esc_poly = self.roi_escaparate.reshape((-1, 1, 2))
                p1_poly = None
                if hasattr(self, 'roi_prueba1') and self.roi_prueba1 is not None:
                    p1_poly = self.roi_prueba1.reshape((-1, 1, 2))
                p2_poly = None
                if hasattr(self, 'roi_prueba2') and self.roi_prueba2 is not None:
                    p2_poly = self.roi_prueba2.reshape((-1, 1, 2))
                cosmetic_polys = [p for p in (esc_poly, roi_poly, p1_poly, p2_poly) if p is not None]

                detections = []
                staff_detected = 0

                # ── Modelo 1: amazonas (interacciones + estante) ──
                # Usamos TODAS las clases del modelo (0..15 para cosmeticos).
                # Si self.model es None (cosmetics_enabled=False) -> None.
                track_cls = None
                if self.model is not None and hasattr(self.model, 'names') \
                        and self.model.names:
                    track_cls = list(self.model.names.keys())

                # Buffer de candidatos cosmeticos: lo declaramos antes para
                # que el resto del pipeline funcione aun con cosmetics_enabled=False.
                cosmetic_candidates = []

                # Gate global: si el cliente desactivo el toggle "Cosmeticos",
                # saltamos toda la inferencia del modelo de productos. La
                # deteccion de personas (modelo 2) y la analitica de
                # genero/edad NO se ven afectadas.
                if not self.cosmetics_enabled:
                    results_amz = None
                    results_amz_alt = None
                    if self.frame_counter % 60 == 0:
                        print("⏸️  Cosmeticos OFF (toggle del cliente) — modelo de productos saltado")

                _run_cosmetics = self.cosmetics_enabled

                # Debug: confirmar que el frame tiene dimensiones razonables
                try:
                    _h, _w = inference_image.shape[:2]
                    if self.frame_counter % 30 == 0:
                        print(f"🖼️ Frame {self.frame_counter}: {_w}x{_h} cls_activas={len(track_cls) if track_cls else 0} conf={self.confidence_threshold} imgsz=1280")
                        # Guardar el frame crudo para inspeccion manual.
                        try:
                            import os as _os
                            _dbg_dir = _os.path.join(
                                self._PROJECT_ROOT, 'output', 'debug_cosmeticos'
                            )
                            _os.makedirs(_dbg_dir, exist_ok=True)
                            _dbg_path = _os.path.join(_dbg_dir, f"frame_{self.frame_counter:06d}.jpg")
                            cv2.imwrite(_dbg_path, inference_image)
                            print(f"💾 Frame guardado: {_dbg_path}")
                        except Exception as _e:
                            print(f"⚠️ No pude guardar frame debug: {_e}")
                except Exception:
                    pass

                # Conf cosmeticos: piso adaptativo. Si NO hay cosmeticos
                # trackeados aun, usamos 0.25 para evitar falsos positivos.
                # Si YA hay tracks activos, bajamos a 0.10 para no perderlos
                # cuando el cliente los agarra (oclusion parcial baja la conf).
                _has_active_cosmetics = any(
                    t.get('class', -1) != -1
                    for t in self.active_tracks.values()
                )
                if _has_active_cosmetics:
                    _cos_conf = max(0.05, float(self.confidence_threshold))
                else:
                    _cos_conf = max(0.20, float(self.confidence_threshold))
                if _run_cosmetics:
                    results_amz = self.model.track(
                        inference_image,
                        imgsz=1280,
                        conf=_cos_conf,
                        iou=self.iou_threshold,
                        classes=track_cls,
                        verbose=False,
                        max_det=100,
                        persist=True,
                        tracker="botsort.yaml",
                    )

                    # TTA segunda pasada a imgsz=960 con conf algo menor para
                    # capturar SKUs que la inferencia a 1280 perdio. Las
                    # detecciones se acumulan y luego se filtran por overlap
                    # con las de 1280 para evitar duplicados.
                    results_amz_alt = None
                    if self.frame_counter % 2 == 0:  # cada 2 frames para latencia
                        try:
                            results_amz_alt = self.model.predict(
                                inference_image,
                                imgsz=960,
                                conf=max(0.18, _cos_conf - 0.05),
                                iou=self.iou_threshold,
                                classes=track_cls,
                                verbose=False,
                                max_det=80,
                            )
                        except Exception as _e:
                            results_amz_alt = None

                # Debug: cuantas detecciones crudas devolvio el modelo cosmeticos
                if results_amz and results_amz[0].boxes is not None:
                    _nraw = len(results_amz[0].boxes)
                    if _nraw > 0:
                        try:
                            _cls_raw = results_amz[0].boxes.cls.cpu().numpy().astype(int).tolist()
                            _conf_raw = results_amz[0].boxes.conf.cpu().numpy().tolist()
                            print(f"🔍 Cosmeticos RAW ({_nraw}): cls={_cls_raw} conf={[round(c,2) for c in _conf_raw]}")
                        except Exception:
                            pass
                    elif self.frame_counter % 30 == 0:
                        print(f"🔍 Cosmeticos RAW: 0 detecciones (frame {self.frame_counter})")

                # Buffer de candidatos cosmeticos ya fue declarado arriba
                # (antes del gate cosmetics_enabled). Aqui solo se llena.

                if results_amz and results_amz[0].boxes is not None:
                    det = results_amz[0].boxes
                    boxes = det.xyxy.cpu().numpy()
                    cls = det.cls.cpu().numpy()
                    confs = det.conf.cpu().numpy() if det.conf is not None else [0.5] * len(boxes)
                    track_ids = det.id.int().cpu().numpy() if det.id is not None else None

                    fh_img, fw_img = inference_image.shape[:2]
                    # Cap muy permisivo: SKU acercado a la camara puede
                    # ocupar hasta ~70% del frame. Solo descartamos si
                    # cubre el frame entero (claramente falso positivo).
                    max_cos_w = fw_img * 0.70
                    max_cos_h = fh_img * 0.80

                    for i in range(boxes.shape[0]):
                        staff_id = int(cls[i])
                        if staff_id not in self.staff_names:
                            continue
                        box = boxes[i]
                        center = self.center_of(box)
                        confidence = confs[i] if i < len(confs) else 0.5

                        tid = int(track_ids[i]) if track_ids is not None else None
                        # Si el SKU ya estaba siendo trackeado, relajamos los
                        # filtros de tamano/aspect (puede estar en la mano,
                        # ladeado, oculto parcialmente).
                        is_tracked_already = (
                            tid is not None
                            and (tid in self._product_track_ids
                                 or tid in self.active_tracks)
                        )

                        bw_cos = float(box[2] - box[0])
                        bh_cos = float(box[3] - box[1])
                        # Solo descartamos SKUs que ocupan casi todo el frame
                        # (>70% W o >80% H) - claros falsos positivos.
                        # El aspect ratio NO se filtra: al agarrar el producto
                        # puede quedar rotado/horizontal.
                        if not is_tracked_already:
                            if bw_cos > max_cos_w or bh_cos > max_cos_h:
                                continue

                        # Sin filtro de ROI: el cliente puede mostrar el
                        # producto en cualquier parte del frame (estante,
                        # mano, frente a camara). Si el modelo lo detecta
                        # con confianza suficiente, lo aceptamos.

                        cosmetic_candidates.append({
                            'class': staff_id,
                            'box': box,
                            'center': center,
                            'confidence': confidence,
                            'track_id': tid,
                        })

                # Mergear detecciones del TTA alterno (imgsz=960). Solo se
                # agregan las que NO se solapan (IoU<=0.40) con ninguna del
                # pase principal — asi rescatamos SKUs perdidos sin duplicar.
                if results_amz_alt and results_amz_alt[0].boxes is not None:
                    det_alt = results_amz_alt[0].boxes
                    boxes_alt = det_alt.xyxy.cpu().numpy()
                    cls_alt = det_alt.cls.cpu().numpy()
                    confs_alt = (det_alt.conf.cpu().numpy()
                                 if det_alt.conf is not None
                                 else [0.5] * len(boxes_alt))

                    fh_img2, fw_img2 = inference_image.shape[:2]
                    max_cos_w2 = fw_img2 * 0.25
                    max_cos_h2 = fh_img2 * 0.40

                    for i in range(boxes_alt.shape[0]):
                        staff_id_a = int(cls_alt[i])
                        if staff_id_a not in self.staff_names:
                            continue
                        box_a = boxes_alt[i]
                        bw_a = float(box_a[2] - box_a[0])
                        bh_a = float(box_a[3] - box_a[1])
                        if bw_a > max_cos_w2 or bh_a > max_cos_h2:
                            continue
                        if bh_a > 0 and (bw_a / bh_a) > 1.4:
                            continue

                        center_a = self.center_of(box_a)
                        # ROI filter
                        if cosmetic_polys:
                            inside_any = any(
                                cv2.pointPolygonTest(
                                    poly, (int(center_a[0]), int(center_a[1])), False
                                ) >= 0
                                for poly in cosmetic_polys
                            )
                            if not inside_any:
                                continue

                        # Descartar si ya hay un candidato del pase principal
                        # cubriendo esta zona (mismo SKU o no, IoU>0.40)
                        is_dup = False
                        for c in cosmetic_candidates:
                            if self.calculate_iou(box_a, c['box']) > 0.40:
                                is_dup = True
                                break
                        if is_dup:
                            continue

                        confidence_a = (confs_alt[i] if i < len(confs_alt)
                                        else 0.4)
                        cosmetic_candidates.append({
                            'class': staff_id_a,
                            'box': box_a,
                            'center': center_a,
                            'confidence': float(confidence_a),
                            'track_id': None,  # BoTSORT lo asigna proximo frame
                        })

                # ── Modelo 2: personas (YOLO estándar, clase 0 COCO) ──
                # Buffer de bboxes de personas para descartar cosmeticos que
                # se solapan con ellas (falsos positivos).
                person_boxes_for_filter = []

                if self.person_model is not None:
                    # Conf personas: piso 0.30 para no perder personas a la
                    # distancia. El modelo COCO de YOLO es robusto, no genera
                    # falsos positivos a 0.30.
                    _per_conf = max(0.30, float(self.confidence_threshold))
                    results_per = self.person_model.track(
                        inference_image,
                        imgsz=640,
                        conf=_per_conf,
                        iou=self.iou_threshold,
                        classes=[0],
                        verbose=False,
                        max_det=50,
                        persist=True,
                        tracker="botsort.yaml",
                    )

                    if results_per and results_per[0].boxes is not None:
                        det_p = results_per[0].boxes
                        boxes_p = det_p.xyxy.cpu().numpy()
                        confs_p = det_p.conf.cpu().numpy() if det_p.conf is not None else [0.5] * len(boxes_p)
                        track_ids_p = det_p.id.int().cpu().numpy() if det_p.id is not None else None

                        for i in range(boxes_p.shape[0]):
                            box = boxes_p[i]
                            center = self.center_of(box)
                            confidence = confs_p[i] if i < len(confs_p) else 0.5

                            # Filtro ROI
                            if roi_poly is not None:
                                if cv2.pointPolygonTest(roi_poly, (int(center[0]), int(center[1])), False) < 0:
                                    continue

                            # Filtro de aspecto: personas son más altas que anchas
                            bw = box[2] - box[0]
                            bh = box[3] - box[1]
                            if bh < bw * 0.8:
                                continue

                            # Offset de track_id para no colisionar con amazonas
                            tid_p = int(track_ids_p[i]) + 100000 if track_ids_p is not None else None
                            detections.append({
                                'class': -1,  # -1 = persona genérica
                                'box': box,
                                'center': center,
                                'confidence': confidence,
                                'track_id': tid_p,
                            })
                            staff_detected += 1
                            person_boxes_for_filter.append(box)

                # ── Pase DEDICADO: cosmeticos sobre el cuerpo ──
                # Por cada persona detectada, hacemos un crop con padding
                # 30% y corremos el modelo cosmeticos sobre ese crop a
                # imgsz=640. Como el crop es chico, el SKU sobre el cuerpo
                # queda con muchos mas pixeles relativos -> el modelo lo
                # detecta donde antes no podia (oclusion parcial / fondo
                # distinto al training). Conf super baja (0.05) porque
                # ya sabemos que es zona de interes (la persona esta ahi).
                if person_boxes_for_filter and _run_cosmetics:
                    fh_full, fw_full = inference_image.shape[:2]
                    for pb in person_boxes_for_filter:
                        try:
                            px1, py1, px2, py2 = [int(v) for v in pb]
                            pw = px2 - px1
                            ph = py2 - py1
                            if pw < 40 or ph < 80:
                                continue
                            pad_x = int(pw * 0.30)
                            pad_y = int(ph * 0.30)
                            cx1 = max(0, px1 - pad_x)
                            cy1 = max(0, py1 - pad_y)
                            cx2 = min(fw_full, px2 + pad_x)
                            cy2 = min(fh_full, py2 + pad_y)
                            person_crop = inference_image[cy1:cy2, cx1:cx2]
                            if person_crop.size == 0:
                                continue

                            res_crop = self.model.predict(
                                person_crop,
                                imgsz=640,
                                conf=0.05,
                                iou=self.iou_threshold,
                                classes=track_cls,
                                verbose=False,
                                max_det=20,
                            )
                            if not res_crop or res_crop[0].boxes is None:
                                continue
                            cb_det = res_crop[0].boxes
                            cboxes = cb_det.xyxy.cpu().numpy()
                            ccls = cb_det.cls.cpu().numpy()
                            cconfs = (cb_det.conf.cpu().numpy()
                                      if cb_det.conf is not None
                                      else [0.5] * len(cboxes))

                            for k in range(cboxes.shape[0]):
                                staff_id_k = int(ccls[k])
                                if staff_id_k not in self.staff_names:
                                    continue
                                bx1, by1, bx2, by2 = cboxes[k]
                                # Mapear coords del crop al frame completo
                                gx1 = bx1 + cx1
                                gy1 = by1 + cy1
                                gx2 = bx2 + cx1
                                gy2 = by2 + cy1
                                gbox = np.array([gx1, gy1, gx2, gy2])
                                gcenter = self.center_of(gbox)
                                gconf = (cconfs[k] if k < len(cconfs)
                                         else 0.4)

                                # Descartar si ya hay un candidato con
                                # IoU>0.40 (duplicado del pase principal).
                                is_dup = False
                                for c in cosmetic_candidates:
                                    if self.calculate_iou(gbox, c['box']) > 0.40:
                                        is_dup = True
                                        break
                                if is_dup:
                                    continue

                                cosmetic_candidates.append({
                                    'class': staff_id_k,
                                    'box': gbox,
                                    'center': gcenter,
                                    'confidence': float(gconf),
                                    'track_id': None,
                                    'source': 'person_crop',
                                })
                        except Exception as _e:
                            logger.debug(f"person-crop SKU pass error: {_e}")

                # ── Pase de RECOVERY para SKUs en mano ──
                # Para cada SKU en _held_skus que NO fue detectado este
                # frame, hacemos un crop generoso (2.0x) alrededor de su
                # ultima posicion conocida y corremos el modelo a conf
                # muy baja con flip horizontal. Esto evita los "blinks"
                # cuando el cliente rota/oclude el producto en su mano.
                if self._held_skus and _run_cosmetics:
                    fh_full3, fw_full3 = inference_image.shape[:2]
                    # tids ya cubiertos por los pases anteriores
                    detected_tids_this_frame = {
                        c.get('track_id') for c in cosmetic_candidates
                        if c.get('track_id') is not None
                    }
                    for sku_tid_held, info in list(self._held_skus.items()):
                        if sku_tid_held in detected_tids_this_frame:
                            continue
                        last_box = info.get('last_box')
                        sku_class_held = info.get('sku_class', -1)
                        if last_box is None or sku_class_held < 0:
                            continue
                        try:
                            lx1, ly1, lx2, ly2 = [float(v) for v in last_box]
                            lw = max(1.0, lx2 - lx1)
                            lh = max(1.0, ly2 - ly1)
                            # Crop de 2.0x el bbox original
                            pad_x_h = lw * 1.0
                            pad_y_h = lh * 1.0
                            rx1 = int(max(0, lx1 - pad_x_h))
                            ry1 = int(max(0, ly1 - pad_y_h))
                            rx2 = int(min(fw_full3, lx2 + pad_x_h))
                            ry2 = int(min(fh_full3, ly2 + pad_y_h))
                            recover_crop = inference_image[ry1:ry2, rx1:rx2]
                            if recover_crop.size == 0:
                                continue
                            # Si el holder esta vivo, expandir hacia su torso
                            holder_tid_h = info.get('person_tid')
                            if holder_tid_h in self.active_tracks:
                                pb_h = self.active_tracks[holder_tid_h].get('box')
                                if pb_h is not None:
                                    rx1 = int(max(0, min(rx1, pb_h[0])))
                                    ry1 = int(max(0, min(ry1, pb_h[1])))
                                    rx2 = int(min(fw_full3, max(rx2, pb_h[2])))
                                    ry2 = int(min(fh_full3, max(ry2, pb_h[3])))
                                    recover_crop = inference_image[ry1:ry2, rx1:rx2]
                                    if recover_crop.size == 0:
                                        continue

                            # Pase normal + pase con flip horizontal
                            crops_to_try = [(recover_crop, False)]
                            try:
                                crops_to_try.append((cv2.flip(recover_crop, 1), True))
                            except Exception:
                                pass

                            for crop_img, is_flipped in crops_to_try:
                                try:
                                    res_rec = self.model.predict(
                                        crop_img,
                                        imgsz=640,
                                        conf=0.03,
                                        iou=self.iou_threshold,
                                        classes=[sku_class_held],
                                        verbose=False,
                                        max_det=10,
                                    )
                                except Exception:
                                    continue
                                if (not res_rec or res_rec[0].boxes is None
                                        or len(res_rec[0].boxes) == 0):
                                    continue
                                rb_det = res_rec[0].boxes
                                rboxes = rb_det.xyxy.cpu().numpy()
                                rcls = rb_det.cls.cpu().numpy()
                                rconfs = (rb_det.conf.cpu().numpy()
                                          if rb_det.conf is not None
                                          else [0.3] * len(rboxes))
                                # Tomar la mas confiada de la clase esperada
                                best_idx = -1
                                best_c = -1.0
                                for j in range(rboxes.shape[0]):
                                    if int(rcls[j]) != sku_class_held:
                                        continue
                                    if float(rconfs[j]) > best_c:
                                        best_c = float(rconfs[j])
                                        best_idx = j
                                if best_idx < 0:
                                    continue
                                bx1, by1, bx2, by2 = rboxes[best_idx]
                                # Si fue flipped, invertir en X
                                ch_w = float(crop_img.shape[1])
                                if is_flipped:
                                    bx1f = ch_w - bx2
                                    bx2f = ch_w - bx1
                                    bx1, bx2 = bx1f, bx2f
                                # Mapear al frame completo
                                gx1 = bx1 + rx1
                                gy1 = by1 + ry1
                                gx2 = bx2 + rx1
                                gy2 = by2 + ry1
                                gbox_h = np.array([gx1, gy1, gx2, gy2])
                                gcenter_h = self.center_of(gbox_h)
                                cosmetic_candidates.append({
                                    'class': sku_class_held,
                                    'box': gbox_h,
                                    'center': gcenter_h,
                                    'confidence': best_c,
                                    'track_id': sku_tid_held,
                                    'source': 'held_recovery',
                                })
                                detected_tids_this_frame.add(sku_tid_held)
                                if self.debug_mode:
                                    print(
                                        f"♻️ Recovery SKU#{sku_tid_held} "
                                        f"({'flip' if is_flipped else 'orig'}) "
                                        f"conf={best_c:.2f}"
                                    )
                                break  # ya recuperado, no probar mas variantes
                        except Exception as _e:
                            logger.debug(f"held-recovery error: {_e}")

                # Aceptar TODOS los cosmeticos que sobrevivieron a los
                # filtros previos (size cap). NO descartamos por overlap
                # con persona: ahora el ROI de persona y el ROI del SKU
                # pueden coexistir, asi el cliente puede tener el producto
                # en la mano, contra el cuerpo o cerca de la camara y
                # ambos siguen siendo trackeados independientemente.
                final_cosmetics = 0
                for cand in cosmetic_candidates:
                    cand_tid = cand.get('track_id')
                    is_tracked = (
                        cand_tid is not None
                        and (cand_tid in self._product_track_ids
                             or cand_tid in self.active_tracks)
                    )
                    detections.append(cand)
                    staff_detected += 1
                    final_cosmetics += 1
                    try:
                        print(
                            f"🛒 Cosmetico {'(tracked) ' if is_tracked else ''}"
                            f"aceptado: id={cand['class']} "
                            f"name={self.staff_names.get(cand['class'], '?')} "
                            f"conf={float(cand['confidence']):.2f} "
                            f"tid={cand_tid}"
                        )
                    except Exception:
                        pass

                if self.debug_mode and detections:
                    print(f"📊 Frame {self.frame_counter}: {len(detections)} detectados ({staff_detected} con track)")

                # Actualizar tracks usando IDs de BoTSORT (estables con Kalman)
                self._update_tracks_botsort(detections)

                # ── Detectar evento PICKUP (persona toma producto) ──
                # Solo tiene sentido si el modelo de cosmeticos esta activo.
                if _run_cosmetics:
                    self._detect_pickup_events(image)

                # Procesar entrada/salida
                persons_inside = self.process_entry_exit_logic(image)

                # Verificar alertas periódicas
                self.check_periodic_alerts(image)

                # ── Modulos de analitica retail ──
                self._run_analytics(image)

                # Log periódico
                if self.frame_counter % 30 == 0:
                    if self.debug_mode:
                        print(f"\n📈 Resumen Frame {self.frame_counter}:")
                        print(f"   Empleados en ROI: {persons_inside}")
                        print(f"   Total en área: {self.personas_en_area}")
                        print(f"   Tracks activos: {len(self.active_tracks)}")

                        # Mostrar contadores por empleado
                        for staff_id, count in self.employee_counters.items():
                            if count > 0:
                                staff_name = self.get_staff_display_name(staff_id)
                                print(f"   {staff_name}: {count} veces")

                # Dibujar resultados (sobre el estado por cámara)
                processed_image = self.draw_detections(image.copy(), persons_inside)

                # Metadatos
                ec = getattr(self, 'entry_counts', {})
                abc = getattr(self, 'active_by_category', {})
                metadata = {
                    'frame_number': self.frame_counter,
                    'roi_active': activate_roi,
                    'staff_detected': staff_detected,
                    'persons_inside': persons_inside,
                    'persons_in_area': self.personas_en_area,
                    'active_tracks': len(self.active_tracks),
                    'employee_counters': dict(self.employee_counters),
                    'entry_counts': dict(ec),
                    'active_by_category': dict(abc),
                    'total_entries': sum(ec.values()),
                }

                # Alertas pickup -> el cliente las pinta en "Toma de Orden"
                if self._pending_pickup_alerts:
                    metadata['alerts'] = list(self._pending_pickup_alerts)
                    self._pending_pickup_alerts.clear()

                # Enriquecer metadata con analitica retail
                if hasattr(self, '_people_counter'):
                    metadata['people_counter'] = self._people_counter.get_stats()
                if hasattr(self, '_demographics'):
                    metadata['demographics'] = self._demographics.get_counts()
                if hasattr(self, '_attendance_tracker'):
                    metadata['attendance'] = self._attendance_tracker.get_stats()
                if hasattr(self, '_seller_efficiency'):
                    metadata['seller_efficiency'] = self._seller_efficiency.get_stats()
                    award = self._seller_efficiency.get_active_award()
                    if award:
                        metadata['active_award'] = award
                if hasattr(self, '_stock_monitor'):
                    metadata['stock'] = self._stock_monitor.get_stats()

                return processed_image, metadata
            except Exception as e:
                logger.error(f"Error procesando frame: {e}")
                import traceback
                traceback.print_exc()
                return image, {'error': str(e)}
            finally:
                # Guardar último frame procesado en el estado de la cámara y restaurar estado global
                try:
                    cam_state['last_processed_frame'] = self.last_processed_frame
                    cam_state['personas_en_area'] = getattr(self, 'personas_en_area', cam_state.get('personas_en_area', 0))
                    cam_state['entry_counts'] = getattr(self, 'entry_counts', cam_state.get('entry_counts', {}))
                except Exception:
                    pass
                # Restaurar estado global
                self._pop_state()

    def get_active_tracks(self, camera_id: Any = 1) -> Dict[int, Dict[str, Any]]:
        """Tracks activos de una camara, tras procesar un frame.

        process_frame() intercambia el estado global por el de la camara
        al entrar (_push_state) y lo restaura al salir (_pop_state, en un
        finally). Por eso, cuando process_frame() retorna,
        self.active_tracks vuelve a ser el diccionario vacio del __init__
        y los tracks reales quedan en camera_states.

        Sin este accesor, quien quiera leer los tracks desde fuera tiene
        que descubrir ese detalle por su cuenta -- y leer self.active_tracks
        devuelve un diccionario vacio sin dar ningun error.
        """
        return self.camera_states.get(camera_id, {}).get('active_tracks', {})

    def set_roi(self, roi_points: List[Tuple[int, int]]):
        self.roi_polygon = np.array(roi_points, np.int32)
        print(f"✅ ROI actualizado a {len(roi_points)} puntos")

    def reset_counter(self):
        self.personas_en_area = 0
        self.employee_counters.clear()
        self.entry_counts = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}
        self.last_counted_frame = 0
        self.last_counted_id = 0
        self.counted_tracks.clear()
        self.recent_counted_persons.clear()
        self.person_cooldown.clear()
        self.active_tracks.clear()
        self.track_history.clear()
        self.movement_history.clear()
        self.sent_entry_photos.clear()
        self.sent_exit_photos.clear()
        self.last_sent_time.clear()
        self.alert_minutes_sent.clear()
        # Reiniciar modulos de analitica
        if hasattr(self, '_demographics'):
            self._demographics.reset()
        if hasattr(self, '_people_counter'):
            self._people_counter.reset()
        if hasattr(self, '_attendance_tracker'):
            self._attendance_tracker.reset()
        if hasattr(self, '_seller_efficiency'):
            self._seller_efficiency.reset()
        if hasattr(self, '_stock_monitor'):
            self._stock_monitor.reset()
        print("🔄 Contadores reiniciados")

    def toggle_minimal_info(self):
        self.show_minimal_info = not self.show_minimal_info
        status = "MÍNIMA" if self.show_minimal_info else "COMPLETA"
        print(f"🔧 Información: {status}")

    def get_stats(self) -> Dict[str, Any]:
        persons_inside = 0
        
        for track in self.active_tracks.values():
            roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
            is_inside = self.is_inside_polygon(track['center'], roi_polygon_points)
            
            if is_inside:
                persons_inside += 1
        
        ec = getattr(self, 'entry_counts', {})
        stats = {
            'total_persons_in_area': self.personas_en_area,
            'persons_inside': persons_inside,
            'frame_counter': self.frame_counter,
            'active_tracks': len(self.active_tracks),
            'employee_counters': dict(self.employee_counters),
            'entry_counts': dict(ec),
            'total_entries': sum(ec.values()),
            'staff_names': dict(self.staff_names),
            'roi_points': self.roi_polygon.tolist(),
        }
        # Agregar stats de analitica retail
        if hasattr(self, '_people_counter'):
            stats['people_counter'] = self._people_counter.get_stats()
        if hasattr(self, '_demographics'):
            stats['demographics'] = self._demographics.get_counts()
        if hasattr(self, '_attendance_tracker'):
            stats['attendance'] = self._attendance_tracker.get_stats()
        if hasattr(self, '_seller_efficiency'):
            stats['seller_efficiency'] = self._seller_efficiency.get_stats()
        if hasattr(self, '_stock_monitor'):
            stats['stock'] = self._stock_monitor.get_stats()
        return stats

    # ── Orquestacion de modulos de analitica retail ──────────────

    def _run_analytics(self, frame: np.ndarray):
        """Ejecuta todos los modulos de analitica en cada frame."""
        try:
            # 1. Actualizar contador de personas activas
            if hasattr(self, '_people_counter'):
                self._people_counter.update_active(set(self.active_tracks.keys()))

            # 2. Actualizar tracker de atencion (proximidad vendedor-cliente)
            if hasattr(self, '_attendance_tracker') and self.active_tracks:
                new_attendances = self._attendance_tracker.update(self.active_tracks)
                for att in new_attendances:
                    # Registrar en eficiencia del vendedor
                    if hasattr(self, '_seller_efficiency'):
                        self._seller_efficiency.record_interaction(
                            att['seller_id'], att['client_id'], att['duration']
                        )
                    # Log
                    if hasattr(self, '_analytics_logger'):
                        self._analytics_logger.log_person_attended(
                            att['seller_id'], att['client_id'], att['duration']
                        )

            # 3. Verificar premio de eficiencia horario
            if hasattr(self, '_seller_efficiency'):
                total_unique = self._people_counter.total_unique if hasattr(self, '_people_counter') else 0
                award = self._seller_efficiency.check_award(total_unique)
                if award and hasattr(self, '_analytics_logger'):
                    self._analytics_logger.log_efficiency_award(
                        award['seller_id'], award['count'],
                        award['avg_time'], award['total_people']
                    )

            # 4. Monitoreo de stock (solo si hay ROIs configurados)
            if hasattr(self, '_stock_monitor'):
                stock_alerts = self._stock_monitor.update(frame)
                for alert in stock_alerts:
                    if hasattr(self, '_analytics_logger'):
                        self._analytics_logger.log_stock_alert(
                            alert['roi_name'], alert['status'], alert['fill_ratio']
                        )

        except Exception as e:
            logger.error(f"Error en analytics: {e}")

    def configure_sellers(self, seller_ids: list):
        """Configura IDs de vendedores desde el exterior.

        Args:
            seller_ids: Lista de track_ids que son vendedores
        """
        if hasattr(self, '_attendance_tracker'):
            self._attendance_tracker.set_seller_ids(seller_ids)
            for sid in seller_ids:
                self._seller_efficiency.register_seller(sid)

    def add_product_roi(self, x1: int, y1: int, x2: int, y2: int, name: str):
        """Agrega un ROI de producto para monitoreo de stock."""
        if hasattr(self, '_stock_monitor'):
            self._stock_monitor.add_roi(x1, y1, x2, y2, name)

    def set_stock_reference(self, frame: np.ndarray, roi_index: int):
        """Captura imagen de referencia (estante lleno) para un ROI de stock."""
        if hasattr(self, '_stock_monitor'):
            self._stock_monitor.set_reference(frame, roi_index)

    def get_analytics_stats(self) -> Dict[str, Any]:
        """Retorna estadisticas completas de todos los modulos de analitica."""
        stats = {}
        if hasattr(self, '_people_counter'):
            stats['people_counter'] = self._people_counter.get_stats()
        if hasattr(self, '_demographics'):
            stats['demographics'] = self._demographics.get_counts()
        if hasattr(self, '_attendance_tracker'):
            stats['attendance'] = self._attendance_tracker.get_stats()
        if hasattr(self, '_seller_efficiency'):
            stats['seller_efficiency'] = self._seller_efficiency.get_stats()
        if hasattr(self, '_stock_monitor'):
            stats['stock'] = self._stock_monitor.get_stats()
        return stats

    def _update_track_class_stability(self, track_id: int):
        """Compute majority vote over recent class observations and update track class if stable."""
        history = self.track_class_history.get(track_id)
        if not history:
            return

        counts = Counter(history)
        most_common, count = counts.most_common(1)[0]
        if len(history) >= 1:
            proportion = count / len(history)
            # require minimum number of observations or proportion
            min_obs = min(3, self.class_history_len)
            if len(history) >= min_obs and proportion >= self.class_stability_threshold:
                # update track's class to stable one
                if track_id in self.active_tracks:
                    self.active_tracks[track_id]['class'] = most_common
                    self.active_tracks[track_id]['stable_class'] = most_common


def create_staff_processor(**kwargs) -> PersonAmazonas:
    return PersonAmazonas(**kwargs)