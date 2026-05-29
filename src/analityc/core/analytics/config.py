"""
analytics/config.py - Parametros configurables para analitica retail.
"""


class AnalyticsConfig:
    """Contenedor de parametros configurables para todos los modulos de analitica."""

    # ── Distancia minima vendedor-cliente para considerar atencion (pixeles) ──
    ATTENDANCE_DISTANCE_PX: int = 150

    # ── Tiempo minimo de proximidad para validar atencion (segundos) ──
    ATTENDANCE_MIN_TIME_SEC: float = 3.0

    # ── Intervalo de premio eficiencia (segundos) ──
    EFFICIENCY_AWARD_INTERVAL_SEC: int = 3600

    # ── Umbrales de stock ──
    STOCK_THRESHOLD_LOW: float = 0.30
    STOCK_THRESHOLD_OK: float = 0.70

    # ── Rangos de edad (min, max) ──
    AGE_RANGES = [
        (0, 12), (13, 17), (18, 25), (26, 35),
        (36, 50), (51, 65), (65, 100),
    ]

    # ── Mapeo de indices del modelo Caffe a rangos propios ──
    # Indices Caffe: 0=(0-2), 1=(4-6), 2=(8-12), 3=(15-20), 4=(25-32), 5=(38-43), 6=(48-53), 7=(60+)
    CAFFE_AGE_TO_RANGE = {
        0: "0-12",   # (0-2)
        1: "0-12",   # (4-6)
        2: "0-12",   # (8-12)
        3: "13-17",  # (15-20) -> aprox
        4: "26-35",  # (25-32)
        5: "36-50",  # (38-43)
        6: "51-65",  # (48-53)
        7: "65+",    # (60+)
    }

    # ── IDs de vendedores (si se asignan manualmente) ──
    SELLER_IDS: list = []

    # ── Color de ropa de vendedor (HSV lower, upper) para deteccion automatica ──
    # Ejemplo: polo rojo -> HSV ~(0,100,100)-(10,255,255)
    SELLER_COLOR_HSV_LOWER = (0, 0, 0)
    SELLER_COLOR_HSV_UPPER = (180, 255, 255)
    SELLER_COLOR_ENABLED: bool = False

    # ── ROIs de producto: lista de (x1, y1, x2, y2, nombre_producto) ──
    PRODUCT_ROIS: list = []

    # ── Directorio de imagenes de referencia para stock ──
    STOCK_REFERENCE_DIR: str = "output/stock_references"

    # ── Parametros de precision MAXIMA del clasificador demografico ──
    # Filosofia: PRECISION SOBRE COBERTURA. Es preferible devolver
    # "Desconocido" a equivocarse en genero/edad. Los thresholds siguientes
    # estan calibrados para ~100% de precision sacrificando recall en
    # personas lejanas/de espaldas/borrosas. Si necesitas mas cobertura
    # (a costa de error), relaja a los valores comentados [PERMISIVO].

    # Voting temporal: acumular N samples de CALIDAD antes de comprometer.
    # Cada sample debe pasar todos los filtros de calidad y pose.
    # 3 es balance entre velocidad (~150ms con cara grande) y robustez.
    # Combinado con fast-commit, una cara clara puede commitar en 1 frame.
    DEMO_MIN_SAMPLES: int = 3                  # [ANTES: 8]
    # Confianza minima del top1 tras voting robusto. Es el threshold
    # BASE para caras grandes (cara real >=120px). Para caras mas chicas
    # se baja proporcionalmente (ver DEMO_ADAPTIVE_THRESHOLD_*).
    DEMO_MIN_CONFIDENCE: float = 0.85          # [PERMISIVO: 0.55]
    # Margen minimo top1 - top2 para comprometer genero.
    DEMO_MIN_MARGIN: float = 0.35              # [PERMISIVO: 0.12]

    # ── Threshold adaptativo por tamano de cara ──
    # Una cara de 30px aporta menos informacion que una de 150px. Exigir
    # 0.85 a una cara chica resulta en NUNCA clasificar personas lejanas
    # aunque el modelo prediga correctamente. Bajamos el threshold para
    # caras chicas manteniendo la garantia de precision para caras grandes.
    DEMO_ADAPTIVE_THRESHOLD: bool = True
    # Tamanos de cara (en px del crop alineado tras upscale a YuNet) y
    # confianza minima asociada. El upscale tipico es ~2x para personas
    # lejanas en CCTV, asi que estos valores son ~2x los px reales.
    # Forma: [(face_w_min, min_conf, min_margin)] ordenado de mayor a
    # menor. Se aplica el primer bracket cuyo umbral cara cumpla.
    # Tradeoff entre cobertura y precision teorica esperada:
    #   0.85 -> precision ~100% (cara grande, modelo SEGURO)
    #   0.72 -> precision ~95%  (cara mediana, modelo moderado)
    #   0.60 -> precision ~85%  (cara chica, mejor que Desconocido)
    # ── Fast-commit: clasificacion INMEDIATA en 1 sample ──
    # Si la primera muestra cumple los siguientes criterios, se compromete
    # el resultado en ese mismo frame (no espera mas samples). Para
    # caras nitidas y frontales esto da resultado en ~150ms.
    DEMO_FAST_COMMIT_ENABLED: bool = True
    DEMO_FAST_COMMIT_MIN_CONFIDENCE: float = 0.92  # confianza altisima
    DEMO_FAST_COMMIT_MIN_MARGIN: float = 0.50      # margen claro
    DEMO_FAST_COMMIT_MIN_FACE_W: float = 50.0      # cara nativa >=50px

    DEMO_ADAPTIVE_BRACKETS = [
        # (face_w_min_NATIVA, min_conf, min_margin)
        # face_w_min es el ancho REAL de la cara en el frame (sin upscale)
        # Con alineamiento ArcFace canonico, la confianza del modelo
        # mejora significativamente, asi que estos thresholds son
        # alcanzables sin sacrificar precision.
        (100, 0.75, 0.30),   # cara grande   (>= 100px nativos)
        ( 60, 0.80, 0.35),   # cara mediana  (60-100px)
        ( 40, 0.78, 0.40),   # cara chica    (40-60px)
        ( 25, 0.85, 0.50),   # cara muy chica (25-40px)
        ( 18, 0.92, 0.65),   # cara minuscula (18-25px) -> exige certeza casi total
        # <18px: rechazado por face-too-small (sin suficiente info real)
    ]
    # Tamano minimo del rostro DETECTADO en pixeles.
    # InsightFace fue entrenado con caras de ~112 px. <40 px = ruido.
    # Bajamos a 40 para CCTV cenital donde caras quedan ~50-90px.
    DEMO_MIN_FACE_SIZE: int = 40               # [PERMISIVO: 14]
    # Varianza minima del Laplaciano (rechaza caras borrosas).
    # 25.0 = exige nitidez razonable, descarta motion blur tipico CCTV.
    DEMO_MIN_BLUR_VAR: float = 25.0            # [PERMISIVO: 3.0]
    # Relacion alto/ancho valida para un rostro frontal.
    DEMO_FACE_ASPECT_MIN: float = 0.85         # [PERMISIVO: 0.45]
    DEMO_FACE_ASPECT_MAX: float = 1.45         # [PERMISIVO: 2.20]
    # Peso de decay por frame viejo (1.0 = sin decay; <1.0 olvida historia)
    DEMO_DECAY: float = 0.985
    # Tras cuantos samples VALIDAS sin converger, marcar como Desconocido.
    # Subimos a 200 porque ahora con el threshold adaptativo las personas
    # lejanas no acumulan samples (la cara no pasa filtros) y solo
    # empezamos a contar cuando se acercan. Damos margen para que el
    # voting converja una vez que entra en rango aceptable.
    DEMO_MAX_SAMPLES_BEFORE_GIVEUP: int = 200  # [PERMISIVO: 300]
    # Confianza minima del face detector DNN (Res10 SSD fallback).
    DEMO_FACE_DETECTOR_CONF: float = 0.70      # [PERMISIVO: 0.28]
    # Resultado PROVISIONAL: DESHABILITADO en modo precision maxima.
    # Solo se muestra etiqueta cuando hay commit final.
    DEMO_PROVISIONAL_MIN_MARGIN: float = 0.30
    DEMO_PROVISIONAL_MIN_CONFIDENCE: float = 0.80
    # Contraste minimo (std del gray) para aceptar el rostro.
    DEMO_MIN_CONTRAST: float = 18.0            # [PERMISIVO: 5.0]
    # Asimetria maxima izquierda/derecha del rostro.
    # En CCTV la iluminacion lateral puede producir asym=80-130 NATURAL.
    # Con alineamiento ArcFace canonico, la cara queda centrada pero
    # las sombras de luz no se eliminan. 135 es permisivo pero el
    # voting+ensemble filtra los falsos positivos.
    DEMO_MAX_ASYMMETRY: float = 135.0          # [PERMISIVO: 50.0]
    # Acuerdo minimo entre samples individuales para commit.
    DEMO_COMMIT_AGREE_RATIO: float = 0.85      # [PERMISIVO: 0.55]
    # Acuerdo para resultado provisional (irrelevante si SHOW_PROVISIONAL=False)
    DEMO_PROVISIONAL_AGREE_RATIO: float = 0.85
    # Mostrar resultado provisional. False = solo committed (recomendado).
    DEMO_SHOW_PROVISIONAL: bool = False
    # ── YuNet face detector ──
    # Score minimo de YuNet para aceptar cara.
    # 0.70 = caras razonablemente nitidas. CCTV bueno -> score >= 0.85.
    DEMO_YUNET_MIN_SCORE: float = 0.70         # [PERMISIVO: 0.25]
    # Distancia inter-ocular minima como fraccion del ancho del bbox.
    # 0.22 fuerza cara semi-frontal (no perfil). En CCTV cenital el
    # eye_ratio tipico de una cara valida cae a 0.25-0.40.
    DEMO_MIN_EYE_FACE_RATIO: float = 0.22      # [PERMISIVO: 0.10]
    # Offset horizontal maximo de la nariz vs centro de ojos (fraccion
    # de la distancia inter-ocular). <0.65 = cabeza ligeramente girada
    # pero clasificable. Caras validas en CCTV cenital dan 0.40-0.60.
    DEMO_MAX_NOSE_OFFSET: float = 0.65         # [PERMISIVO: 1.10]
    # Verbose: imprime motivos de rechazo de cara.
    DEMO_DEBUG_REJECTIONS: bool = True
    # Tamano minimo del bbox de la PERSONA. Personas mas lejanas que esto
    # son rechazadas. En CCTV 1080p personas validas miden 250-600px alto.
    # En frames comprimidos 640x400 quedan en 130-280px.
    DEMO_MIN_PERSON_BBOX_W: int = 60           # [PERMISIVO: 30]
    DEMO_MIN_PERSON_BBOX_H: int = 140          # [PERMISIVO: 80]
    # Tamano objetivo del lado mas largo del crop al pasar a YuNet.
    DEMO_YUNET_UPSCALE_TARGET: int = 800       # [PERMISIVO: 640]

    # Tamano minimo de la cara ORIGINAL (sin upscale) en pixeles del
    # frame original. Esto es lo que realmente determina si hay info
    # suficiente para clasificar genero/edad. Una cara upscaleada de
    # 30x30 a 90x90 sigue siendo 30x30 worth of info (interpolada).
    # InsightFace fue entrenado con caras 112x112, asi que <40px reales
    # da clasificaciones poco fiables (sobre todo en genero).
    DEMO_MIN_NATIVE_FACE_W: int = 40
    # Si la cara nativa es menor que esto, el commit se demora y exige
    # MUCHA mas confianza para evitar errores de overconfidence.
    DEMO_SMALL_FACE_CONF_BOOST: float = 0.10

    # ── Pose: solo caras semi-frontales se clasifican ──
    # CCTV cenital (camara montada arriba mirando abajo) produce pitch
    # alto SIEMPRE aunque la persona mire de frente. Pitch tipico medido
    # en pruebas con frames reales: +30 a +40 grados. Ajustamos arriba.
    # Si tu camara es a nivel ojos, baja pitch_max a 25.
    DEMO_POSE_MAX_YAW_DEG: float = 35.0
    DEMO_POSE_MAX_PITCH_DEG: float = 48.0
    DEMO_POSE_MAX_ROLL_DEG: float = 40.0

    # ── Ensemble multi-modelo ──
    # Si hay >=2 clasificadores disponibles, requerir acuerdo en GENERO
    # para comprometer. Si discrepan -> Desconocido. Cero falsos positivos.
    DEMO_REQUIRE_ENSEMBLE_AGREEMENT: bool = True
    # Diferencia maxima permitida entre predicciones de edad de los
    # modelos (en anos). Si superan esto -> Desconocido para edad.
    DEMO_MAX_AGE_DISAGREEMENT_YEARS: float = 12.0

    # ── Face Re-Identification (ArcFace embeddings) ──
    # Cuando esta activo, cuenta personas UNICAS por embedding biometrico
    # en lugar de por track_id. Previene contar dos veces a la misma
    # persona que entra/sale o cuya tracking se rompio.
    REID_ENABLED: bool = True
    # Filename del modelo de embeddings (ArcFace R50 de buffalo_l)
    REID_MODEL_FILENAME: str = "w600k_r50.onnx"
    # Cosine similarity minimo para considerar misma persona.
    # 0.35 con multi-embedding (guardamos varios embeddings por persona)
    # + pose-strict para registro nuevo = balance que captura mismos
    # individuos en distintos angulos sin generar falsos positivos
    # significativos. Bajar a 0.30 si todavia se duplican; subir a
    # 0.45 si se confunden personas distintas.
    REID_SIMILARITY_THRESHOLD: float = 0.35
    # Maximo de embeddings guardados por persona. Almacenar varios
    # capturando distintos angulos/expresiones permite matchear a la
    # misma persona aun cuando cambia de pose dramaticamente.
    REID_MAX_EMBEDDINGS_PER_PERSON: int = 5
    # Yaw maximo PARA REGISTRAR una persona NUEVA. Mas estricto que el
    # yaw general permitido para clasificacion (DEMO_POSE_MAX_YAW_DEG).
    # Las caras de perfil generan embeddings que NO matchean con
    # frontales aunque sean la misma persona, asi que NO debemos crear
    # nuevos registros desde poses laterales. Sin embargo, las caras
    # de perfil pueden seguir intentando MATCHEAR contra personas ya
    # conocidas (por si tuvieran tambien embeddings de perfil).
    REID_MAX_YAW_FOR_REGISTRATION: float = 20.0
    # Pitch maximo para registrar (mas estricto que el de clasificacion)
    REID_MAX_PITCH_FOR_REGISTRATION: float = 35.0
    # Politica de reset: "never", "daily", "weekly", "manual"
    REID_RESET_POLICY: str = "daily"
    # Path persistente de la base de datos
    REID_DB_PATH: str = "output/person_db/persons.pkl"
    # Threshold de confianza demografica minima para almacenar en la DB.
    # Solo guardamos genero/edad si llego con esta confianza.
    REID_MIN_DEMO_CONFIDENCE: float = 0.60

    # ── MiVOLO (modelo SOTA 2024 opcional) ──
    # Path al modelo MiVOLO ONNX. Si no existe, se ignora silenciosamente.
    # Para activarlo: descargar de https://github.com/WildChlamydia/MiVOLO
    # y convertir a ONNX (model_imdb_cross_person_4.22_99.46.pth.onnx) y
    # colocar en models/classifiers/mivolo.onnx
    MIVOLO_MODEL_FILENAME: str = "mivolo.onnx"
    # Input size esperado por MiVOLO (224x224 con cara+cuerpo concatenados).
    MIVOLO_INPUT_SIZE: int = 224
    # Cuanto pesa MiVOLO vs InsightFace en el promedio (0..1).
    # 0.6 = MiVOLO 60%, InsightFace 40% (MiVOLO es mas preciso).
    MIVOLO_WEIGHT_IN_ENSEMBLE: float = 0.60

    # ── Heavy-TTA ──
    # Numero de variantes TTA por inferencia. Cada variante es ~30ms en
    # CPU. 3 = original + flip + zoom_in -> ~90ms por sample. 6 era
    # demasiado lento para ser inmediato (~180ms por sample).
    DEMO_TTA_VARIANTS: int = 3                 # [ANTES: 6]
    # Si las variantes TTA deben coincidir en argmax(genero) para contar.
    # False = promedia todas, mas permisivo (deja que el voting temporal
    # del accumulator filtre el ruido). True = exige >=80% agreement,
    # mas precision por sample pero rechaza muchas con caras chicas.
    DEMO_TTA_REQUIRE_AGREEMENT: bool = False

    # ── Duracion del banner de premio (segundos) ──
    AWARD_BANNER_DURATION_SEC: float = 10.0

    # ── Archivo de log de premios ──
    EFFICIENCY_AWARDS_LOG: str = "output/efficiency_awards.log"

    # ── Archivo de log de stock ──
    STOCK_ALERTS_LOG: str = "output/stock_alerts.log"

    # ── Archivo de log general (JSONL) ──
    ANALYTICS_LOG_JSONL: str = "output/analytics_log.jsonl"

    @classmethod
    def age_range_label(cls, age_idx: int) -> str:
        """Convierte indice del modelo Caffe a etiqueta de rango de edad."""
        return cls.CAFFE_AGE_TO_RANGE.get(age_idx, "18-25")

    @classmethod
    def age_range_from_value(cls, age_value: int) -> str:
        """Convierte un valor numerico de edad a etiqueta de rango."""
        for lo, hi in cls.AGE_RANGES:
            if lo <= age_value <= hi:
                if hi == 100:
                    return "65+"
                return f"{lo}-{hi}"
        return "18-25"
