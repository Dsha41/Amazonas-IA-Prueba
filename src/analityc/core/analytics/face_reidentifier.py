"""
analytics/face_reidentifier.py - Face re-identification para evitar
recontar a la misma persona multiples veces.

Pipeline:
  1. Cuando un track tiene una cara de buena calidad, se calcula su
     embedding ArcFace 512-dim usando w600k_r50.onnx.
  2. Se busca en la base de datos persistente la persona mas parecida
     usando cosine similarity.
  3. Si la similitud supera el threshold -> reasignar al person_uuid
     existente (incrementar visit_count).
  4. Si no -> crear nuevo person_uuid, guardar embedding y demograficos.
  5. La DB se guarda en disco (pickle) y se carga al inicio.
  6. Reset diario automatico a las 00:00 (configurable).

Modelo:
  w600k_r50.onnx (ArcFace R50, viene en buffalo_l de InsightFace).
  Input: (1, 3, 112, 112) RGB normalizado.
  Output: (1, 512) embedding L2-normalizado (mas o menos).

Storage:
  output/person_db/persons.pkl con dict[uuid] -> dict de metadata.
"""

import os
import time
import uuid
import pickle
import threading
import datetime
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .config import AnalyticsConfig

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity entre dos vectores (sin asumir normalizados)."""
    a_norm = a / max(np.linalg.norm(a), 1e-9)
    b_norm = b / max(np.linalg.norm(b), 1e-9)
    return float(np.dot(a_norm, b_norm))


def _get_embeddings(rec: Dict[str, Any]) -> List[np.ndarray]:
    """Devuelve la lista de embeddings de un record, soportando formato
    nuevo (lista) y legacy (un solo embedding)."""
    if 'embeddings' in rec:
        return rec['embeddings']
    # Formato legacy: un solo embedding
    if 'embedding' in rec:
        return [rec['embedding']]
    return []


def _best_match_similarity(query: np.ndarray, rec: Dict[str, Any]
                           ) -> float:
    """Maxima similitud entre el embedding query y todos los embeddings
    almacenados de una persona. Multi-embedding = mas tolerante a pose."""
    embeddings = _get_embeddings(rec)
    if not embeddings:
        return -1.0
    best = -1.0
    for emb in embeddings:
        sim = _cosine_similarity(query, emb)
        if sim > best:
            best = sim
    return best


def _add_embedding_diverse(rec: Dict[str, Any], new_emb: np.ndarray,
                           max_count: int):
    """Agrega un embedding a la persona manteniendo diversidad.

    Si la lista no esta llena, agrega. Si esta llena, reemplaza el
    embedding mas similar al nuevo (para preservar diversidad). Asi la
    persona termina con embeddings de poses/condiciones DISTINTAS y
    matchea mejor a futuro.
    """
    embeddings = _get_embeddings(rec)
    if len(embeddings) < max_count:
        embeddings.append(new_emb.astype(np.float32))
    else:
        # Calcular similitud del nuevo embedding contra los existentes
        sims = [_cosine_similarity(new_emb, e) for e in embeddings]
        # Reemplazar el mas similar (es el mas redundante)
        replace_idx = int(np.argmax(sims))
        embeddings[replace_idx] = new_emb.astype(np.float32)
    rec['embeddings'] = embeddings
    # Borrar formato legacy si existia
    rec.pop('embedding', None)


class FaceReidentifier:
    """Base de datos persistente de embeddings de personas vistas.

    Maneja un dict {uuid -> persona} donde persona contiene embedding,
    demograficos confirmados, contadores y timestamps. Hilo-seguro
    (lock al modificar la DB) pero NO atomic en disco (guardamos
    asincronicamente para no bloquear el frame loop).

    Reset diario configurable: si la fecha del ultimo guardado no es
    hoy, la DB se limpia al inicializar (logica de "ventana de 24h").
    """

    def __init__(self,
                 embedding_session=None,
                 db_path: Optional[str] = None,
                 similarity_threshold: float = 0.50,
                 reset_policy: str = "daily"):
        """
        Args:
            embedding_session: onnxruntime InferenceSession con w600k_r50.
            db_path: ruta del pickle. Default: output/person_db/persons.pkl
            similarity_threshold: cosine minimo para considerar misma persona.
            reset_policy: "never", "daily", "weekly", "manual".
        """
        self._session = embedding_session
        if self._session is not None:
            try:
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
            except Exception as e:
                logger.error(f"FaceReidentifier: sesion ONNX invalida: {e}")
                self._session = None
                self._input_name = None
                self._output_name = None
        else:
            self._input_name = None
            self._output_name = None

        # Paths
        if db_path is None:
            db_path = os.path.join("output", "person_db", "persons.pkl")
        self._db_path = db_path
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        # Directorio donde se guardan las imagenes de rostros (1 por UUID)
        self._faces_dir = os.path.join(
            os.path.dirname(self._db_path), "faces"
        )
        os.makedirs(self._faces_dir, exist_ok=True)

        # Config
        self._sim_threshold = float(similarity_threshold)
        self._reset_policy = reset_policy
        self._lock = threading.RLock()

        # In-memory DB:
        # {
        #   uuid_str: {
        #     'embedding': np.ndarray(512, float32),  # promedio rolling
        #     'embedding_count': int,
        #     'gender': str | None,
        #     'age_range': str | None,
        #     'age_value': float,
        #     'demo_confidence': float,
        #     'first_seen': float (timestamp),
        #     'last_seen': float,
        #     'visit_count': int,
        #     'track_ids': set(int),  # tracks asociados en esta sesion
        #   }
        # }
        self._db: Dict[str, Dict[str, Any]] = {}
        # Mapping track_id -> person_uuid (solo en memoria, no persistido)
        self._track_to_person: Dict[int, str] = {}
        # Timestamp del ultimo guardado (para reset diario)
        self._last_save_date: Optional[datetime.date] = None
        # Throttle: no guardar mas de cada 5s
        self._last_disk_write: float = 0.0
        self._disk_write_interval: float = 5.0
        # mtime del pickle cuando lo leimos por ultima vez. Si el archivo
        # cambia externamente (ej. webapp borra una persona), detectamos
        # el cambio antes del proximo save y aplicamos la eliminacion en
        # nuestra copia en memoria.
        self._db_file_mtime: float = 0.0
        # Counter sesion (para metricas rapidas)
        self._session_new_count: int = 0
        self._session_reidentified_count: int = 0

        self._load_db()

    @property
    def is_available(self) -> bool:
        return self._session is not None

    @property
    def total_unique_persons(self) -> int:
        with self._lock:
            return len(self._db)

    @property
    def session_new_count(self) -> int:
        return self._session_new_count

    @property
    def session_reidentified_count(self) -> int:
        return self._session_reidentified_count

    # ── Persistencia ─────────────────────────────────────────────

    def _load_db(self):
        """Carga la DB del disco aplicando reset_policy."""
        if not os.path.isfile(self._db_path):
            logger.info(f"FaceReidentifier: DB nueva ({self._db_path})")
            self._last_save_date = datetime.date.today()
            self._db_file_mtime = 0.0
            return
        try:
            self._db_file_mtime = os.path.getmtime(self._db_path)
            with open(self._db_path, 'rb') as f:
                payload = pickle.load(f)
            saved_date = payload.get('date')
            self._db = payload.get('db', {})
            saved_date_obj = (
                datetime.date.fromisoformat(saved_date)
                if isinstance(saved_date, str) else saved_date
            )
            today = datetime.date.today()

            if self._should_reset(saved_date_obj, today):
                logger.info(
                    f"FaceReidentifier: reset por politica "
                    f"({self._reset_policy}). Saved={saved_date_obj} "
                    f"today={today}"
                )
                self._db = {}
                self._last_save_date = today
                self._save_db_now()
            else:
                self._last_save_date = saved_date_obj
                logger.info(
                    f"FaceReidentifier: cargadas {len(self._db)} "
                    f"personas de {self._db_path}"
                )
        except Exception as e:
            logger.error(
                f"FaceReidentifier: error cargando DB: {e}. Empezando vacia."
            )
            self._db = {}
            self._last_save_date = datetime.date.today()

    def _should_reset(self, saved_date: Optional[datetime.date],
                      today: datetime.date) -> bool:
        if saved_date is None:
            return False
        if self._reset_policy == "never":
            return False
        if self._reset_policy == "manual":
            return False
        if self._reset_policy == "daily":
            return saved_date != today
        if self._reset_policy == "weekly":
            # Reset si cruzamos un lunes desde el ultimo guardado
            days_diff = (today - saved_date).days
            if days_diff >= 7:
                return True
            # Si el lunes mas reciente esta entre saved_date+1 y today
            for d in range(1, days_diff + 1):
                check_date = saved_date + datetime.timedelta(days=d)
                if check_date.weekday() == 0:  # Lunes
                    return True
            return False
        return False

    def _sync_external_deletions(self):
        """Detecta cambios externos al archivo y aplica eliminaciones.

        Si el archivo de DB fue modificado externamente (ej. webapp borro
        una persona) desde nuestra ultima escritura, leemos la version
        actual del disco y:
          - Para UUIDs que estaban en disco antes y YA NO ESTAN ahora:
            tambien los quitamos de nuestra memoria.
          - Para UUIDs nuevos que solo existen en nuestra memoria
            (creados despues del cambio externo): los conservamos.

        Esto resuelve la race condition entre el dashboard que borra
        personas y el proceso de analytics que tenia su propia copia.
        """
        if not os.path.isfile(self._db_path):
            return
        try:
            current_mtime = os.path.getmtime(self._db_path)
            if current_mtime <= self._db_file_mtime:
                return  # archivo no cambio desde nuestra ultima lectura

            # Archivo cambio externamente -> leer y aplicar deltas
            with open(self._db_path, 'rb') as f:
                payload = pickle.load(f)
            disk_db = payload.get('db', {})

            disk_uids = set(disk_db.keys())
            memory_uids = set(self._db.keys())

            # Personas que existen en memoria pero ya NO en disco
            # Y cuya last_seen es ANTERIOR al cambio externo -> fueron
            # borradas externamente. Las que tienen last_seen reciente
            # son personas que agregamos despues del borrado externo.
            externally_deleted = []
            for uid in memory_uids - disk_uids:
                last_seen = float(self._db[uid].get('last_seen', 0.0))
                if last_seen < current_mtime:
                    externally_deleted.append(uid)

            for uid in externally_deleted:
                del self._db[uid]
                # Limpiar track mapping
                stale = [t for t, p in self._track_to_person.items()
                         if p == uid]
                for t in stale:
                    del self._track_to_person[t]

            if externally_deleted:
                logger.info(
                    f"FaceReidentifier: detectado cambio externo, "
                    f"removidas {len(externally_deleted)} personas de "
                    f"memoria: {[u[:8] for u in externally_deleted]}"
                )

            self._db_file_mtime = current_mtime
        except Exception as e:
            logger.debug(f"Error sincronizando con disco: {e}")

    def _save_db_now(self):
        """Guarda la DB al disco de forma sincronica (con lock).

        Antes de guardar, sincroniza con el disco para no sobrescribir
        cambios externos (ej. eliminaciones desde el dashboard webapp).
        """
        with self._lock:
            try:
                self._sync_external_deletions()
                payload = {
                    'date': datetime.date.today().isoformat(),
                    'db': self._db,
                    'version': 1,
                }
                tmp_path = self._db_path + '.tmp'
                with open(tmp_path, 'wb') as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, self._db_path)
                self._last_disk_write = time.time()
                self._last_save_date = datetime.date.today()
                # Actualizar mtime tras nuestra escritura
                self._db_file_mtime = os.path.getmtime(self._db_path)
            except Exception as e:
                logger.error(f"FaceReidentifier: error guardando DB: {e}")

    def _save_throttled(self):
        """Guarda al disco si paso el intervalo de throttle."""
        now = time.time()
        if now - self._last_disk_write >= self._disk_write_interval:
            self._save_db_now()

    def force_save(self):
        """Fuerza guardado inmediato (llamar al shutdown del sistema)."""
        self._save_db_now()

    # ── Guardado de imagenes de rostros ──────────────────────────

    def _save_face_image(self, uid: str, face_bgr: np.ndarray):
        """Guarda la imagen del rostro a disco (output/person_db/faces/UID.jpg).

        Se redimensiona a 160x160 para uniformidad visual en el dashboard.
        Si la imagen es muy chica, se interpolan con INTER_CUBIC.
        """
        if face_bgr is None or face_bgr.size == 0:
            return
        try:
            target = 160
            h, w = face_bgr.shape[:2]
            if max(h, w) > target * 2:
                resized = cv2.resize(
                    face_bgr, (target, target),
                    interpolation=cv2.INTER_AREA
                )
            else:
                resized = cv2.resize(
                    face_bgr, (target, target),
                    interpolation=cv2.INTER_CUBIC
                )
            path = os.path.join(self._faces_dir, f"{uid}.jpg")
            cv2.imwrite(path, resized,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            logger.debug(f"Error guardando rostro {uid}: {e}")

    def _compute_face_score(self, face_bgr: np.ndarray) -> float:
        """Score simple para decidir si reemplazar la imagen guardada.

        Combina tamano + nitidez (Laplacian variance). Cuanto mayor el
        score, mejor la imagen visual del rostro.
        """
        if face_bgr is None or face_bgr.size == 0:
            return 0.0
        try:
            h, w = face_bgr.shape[:2]
            size_score = (h * w) / (112.0 * 112.0)
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(blur_var / 200.0, 2.0)
            return float(size_score + blur_score)
        except Exception:
            return 0.0

    def get_face_image_path(self, uid: str) -> Optional[str]:
        """Devuelve la ruta de la imagen guardada o None si no existe."""
        path = os.path.join(self._faces_dir, f"{uid}.jpg")
        return path if os.path.isfile(path) else None

    def get_all_persons(self) -> List[Dict[str, Any]]:
        """Devuelve lista de todas las personas en la DB, lista para
        serializar (sin embedding raw, sin sets).
        """
        with self._lock:
            out = []
            for uid, rec in self._db.items():
                out.append({
                    'uuid': uid,
                    'gender': rec.get('gender') or 'Desconocido',
                    'age_range': rec.get('age_range') or 'Desconocido',
                    'age_value': float(rec.get('age_value', 0.0)),
                    'demo_confidence': float(
                        rec.get('demo_confidence', 0.0)),
                    'first_seen': float(rec.get('first_seen', 0.0)),
                    'last_seen': float(rec.get('last_seen', 0.0)),
                    'visit_count': int(rec.get('visit_count', 1)),
                    'face_image_available':
                        self.get_face_image_path(uid) is not None,
                })
            return out

    # ── Computo de embeddings ────────────────────────────────────

    def compute_embedding(self, face_bgr: np.ndarray
                          ) -> Optional[np.ndarray]:
        """Calcula el embedding 512-dim de una cara alineada.

        Args:
            face_bgr: imagen BGR de cara alineada (cualquier tamano).

        Returns:
            np.ndarray (512,) float32 L2-normalizado o None si falla.
        """
        if self._session is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            # ArcFace espera 112x112 RGB normalizado a [-1, 1]
            resized = cv2.resize(face_bgr, (112, 112),
                                 interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32)
            arr = (arr - 127.5) / 127.5
            # HWC -> CHW -> NCHW
            blob = arr.transpose(2, 0, 1)[None, ...]

            out = self._session.run(
                [self._output_name], {self._input_name: blob}
            )[0][0]
            # Normalizar L2
            norm = np.linalg.norm(out)
            if norm > 0:
                out = out / norm
            return out.astype(np.float32)
        except Exception as e:
            logger.debug(f"FaceReidentifier embedding error: {e}")
            return None

    # ── API publica ──────────────────────────────────────────────

    def identify_or_register(self, track_id: int,
                             face_bgr: np.ndarray,
                             gender: Optional[str] = None,
                             age_range: Optional[str] = None,
                             age_value: float = 0.0,
                             demo_confidence: float = 0.0,
                             face_yaw: Optional[float] = None,
                             face_pitch: Optional[float] = None,
                             ) -> Optional[Tuple[str, bool, Dict[str, Any]]]:
        """Identifica o registra una persona basado en su cara.

        Args:
            track_id: id de tracking del frame actual (interno).
            face_bgr: imagen BGR de cara alineada.
            gender, age_range, age_value, demo_confidence: demograficos
              committeados (opcional).
            face_yaw, face_pitch: angulos de pose (grados). Si se pasan
              y son demasiado laterales, NO crearemos un nuevo registro
              (solo intentaremos matchear contra existentes). Esto
              previene que perfiles generen "personas fantasma".

        Returns:
            (person_uuid, is_new, person_record) o None si:
              - falla embedding
              - no match Y pose es muy lateral (skip registration)
        """
        if self._session is None:
            return None

        # Si ya tenemos un mapping para este track_id, reusar
        if track_id in self._track_to_person:
            uid = self._track_to_person[track_id]
            with self._lock:
                if uid in self._db:
                    rec = self._db[uid]
                    rec['last_seen'] = time.time()
                    # Actualizar demograficos si llegan ahora
                    self._maybe_update_demographics(
                        rec, gender, age_range, age_value, demo_confidence
                    )
                    return uid, False, dict(rec)

        # Calcular embedding
        emb = self.compute_embedding(face_bgr)
        if emb is None:
            return None

        with self._lock:
            # Sincronizar con disco antes de buscar match. Esto detecta
            # cualquier borrado externo (ej. webapp borro a alguien)
            # y evita re-identificar a una persona que el usuario quiso
            # eliminar.
            self._sync_external_deletions()

            # Buscar mejor match: para cada persona, calcular la maxima
            # similitud entre el query y CUALQUIERA de sus embeddings
            # almacenados (multi-embedding storage).
            best_uid = None
            best_sim = -1.0
            for uid, rec in self._db.items():
                sim = _best_match_similarity(emb, rec)
                if sim > best_sim:
                    best_sim = sim
                    best_uid = uid

            if best_uid is not None and best_sim >= self._sim_threshold:
                # Match: re-identificacion
                rec = self._db[best_uid]
                # Agregar nuevo embedding manteniendo diversidad
                _add_embedding_diverse(
                    rec, emb,
                    AnalyticsConfig.REID_MAX_EMBEDDINGS_PER_PERSON
                )
                rec['embedding_count'] = rec.get(
                    'embedding_count', 1) + 1
                rec['last_seen'] = time.time()
                rec['visit_count'] += 1
                rec['track_ids'].add(int(track_id))
                self._maybe_update_demographics(
                    rec, gender, age_range, age_value, demo_confidence
                )
                # Si esta cara nueva tiene mejor score visual, reemplazar
                new_score = self._compute_face_score(face_bgr)
                old_score = float(rec.get('face_score', 0.0))
                if new_score > old_score * 1.10:
                    rec['face_score'] = new_score
                    self._save_face_image(best_uid, face_bgr)
                self._track_to_person[track_id] = best_uid
                self._session_reidentified_count += 1
                logger.info(
                    "Re-IDENT track=%s persona=%s sim=%.3f "
                    "gender=%s visit=%d",
                    track_id, best_uid[:8], best_sim,
                    rec.get('gender'), rec['visit_count']
                )
                self._save_throttled()
                return best_uid, False, dict(rec)

            # No match -> evaluar si crear nueva persona.
            # NO creamos registros nuevos desde caras de perfil porque
            # sus embeddings no matchearan con frontales de la misma
            # persona y generariamos duplicados. Solo registramos
            # personas nuevas con poses semi-frontales.
            if face_yaw is not None:
                max_yaw = AnalyticsConfig.REID_MAX_YAW_FOR_REGISTRATION
                if abs(face_yaw) > max_yaw:
                    logger.debug(
                        f"REID skip register: yaw={face_yaw:.1f} > "
                        f"{max_yaw} (cara muy lateral)"
                    )
                    return None
            if face_pitch is not None:
                max_pitch = AnalyticsConfig.REID_MAX_PITCH_FOR_REGISTRATION
                if abs(face_pitch) > max_pitch:
                    logger.debug(
                        f"REID skip register: pitch={face_pitch:.1f} > "
                        f"{max_pitch}"
                    )
                    return None

            new_uid = str(uuid.uuid4())
            now = time.time()
            new_rec = {
                'embeddings': [emb.astype(np.float32)],
                'embedding_count': 1,
                'gender': gender,
                'age_range': age_range,
                'age_value': float(age_value),
                'demo_confidence': float(demo_confidence),
                'first_seen': now,
                'last_seen': now,
                'visit_count': 1,
                'track_ids': {int(track_id)},
                'face_score': self._compute_face_score(face_bgr),
            }
            self._db[new_uid] = new_rec
            self._track_to_person[track_id] = new_uid
            self._session_new_count += 1
            # Guardar imagen del rostro
            self._save_face_image(new_uid, face_bgr)
            logger.info(
                "NEW persona=%s track=%s gender=%s age=%s "
                "(top sim=%.3f < %.2f)",
                new_uid[:8], track_id, gender, age_range,
                best_sim if best_uid else 0.0, self._sim_threshold
            )
            self._save_throttled()
            return new_uid, True, dict(new_rec)

    def _maybe_update_demographics(self, rec: Dict[str, Any],
                                   gender, age_range,
                                   age_value, demo_confidence):
        """Actualiza demograficos si el nuevo dato es mejor que el viejo.

        Reglas:
         - Si la persona tiene 'manual_override' = True -> NUNCA actualizar
           (el usuario lo corrigio manualmente, debe quedarse asi).
         - Si no hay datos previos -> guardar.
         - Si hay datos previos con mayor demo_confidence -> NO actualizar.
         - Si nuevo demo_confidence > anterior + 0.05 -> actualizar.
        """
        if rec.get('manual_override'):
            return  # Correccion manual del usuario es sticky
        if gender is None and age_range is None:
            return
        old_conf = float(rec.get('demo_confidence', 0.0))
        new_conf = float(demo_confidence)
        if rec.get('gender') in (None, "Desconocido"):
            update = True
        elif new_conf > old_conf + 0.05:
            update = True
        else:
            update = False
        if update:
            if gender is not None:
                rec['gender'] = gender
            if age_range is not None:
                rec['age_range'] = age_range
            rec['age_value'] = float(age_value)
            rec['demo_confidence'] = new_conf

    def get_demographics_for_person(self, track_id: int
                                    ) -> Optional[Dict[str, Any]]:
        """Devuelve los demograficos almacenados para el track_id."""
        if track_id not in self._track_to_person:
            return None
        uid = self._track_to_person[track_id]
        with self._lock:
            if uid not in self._db:
                return None
            r = self._db[uid]
            return {
                'person_uuid': uid,
                'gender': r.get('gender') or 'Desconocido',
                'age_range': r.get('age_range') or 'Desconocido',
                'age_value': r.get('age_value', 0.0),
                'visit_count': r.get('visit_count', 1),
                'demo_confidence': r.get('demo_confidence', 0.0),
            }

    def remove_track(self, track_id: int):
        """Limpia el mapping track->person cuando un track termina.
        NO borra la persona de la DB (puede volver).
        """
        self._track_to_person.pop(track_id, None)

    def find_duplicates(self, min_similarity: float = 0.30,
                        max_pairs: int = 50) -> List[Dict[str, Any]]:
        """Encuentra pares de personas con similitud sospechosa.

        Util para detectar duplicados que el auto-match no capturo
        por estar en la zona gris (similitud entre min_similarity y
        el threshold de auto-match).

        Args:
            min_similarity: similitud minima para incluir como candidato.
            max_pairs: maximo de pares a devolver (los de mayor similitud).

        Returns:
            Lista de dicts: [{uid_a, uid_b, similarity}], ordenado desc.
        """
        with self._lock:
            uids = list(self._db.keys())
            if len(uids) < 2:
                return []

            pairs = []
            for i in range(len(uids)):
                a_uid = uids[i]
                a_embs = _get_embeddings(self._db[a_uid])
                if not a_embs:
                    continue
                for j in range(i + 1, len(uids)):
                    b_uid = uids[j]
                    b_embs = _get_embeddings(self._db[b_uid])
                    if not b_embs:
                        continue
                    # Maxima similitud entre todos los embeddings de A y B
                    best_sim = -1.0
                    for ea in a_embs:
                        for eb in b_embs:
                            s = _cosine_similarity(ea, eb)
                            if s > best_sim:
                                best_sim = s
                    if best_sim >= min_similarity:
                        pairs.append({
                            'uid_a': a_uid,
                            'uid_b': b_uid,
                            'similarity': float(best_sim),
                        })

            # Ordenar por similitud descendente
            pairs.sort(key=lambda p: p['similarity'], reverse=True)
            return pairs[:max_pairs]

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                'total_persons': len(self._db),
                'session_new': self._session_new_count,
                'session_reidentified': self._session_reidentified_count,
                'active_tracks': len(self._track_to_person),
                'persons_with_demographics': sum(
                    1 for r in self._db.values()
                    if r.get('gender') not in (None, 'Desconocido')
                ),
            }

    def reset(self):
        """Limpia la DB en memoria y en disco. Para uso manual."""
        with self._lock:
            self._db.clear()
            self._track_to_person.clear()
            self._session_new_count = 0
            self._session_reidentified_count = 0
            self._save_db_now()
        logger.info("FaceReidentifier: DB reseteada manualmente")

    def delete_person(self, uid: str) -> bool:
        """Borra una persona individual (registro DB + imagen).

        Returns:
            True si se borro, False si el uid no existia.
        """
        with self._lock:
            if uid not in self._db:
                return False
            del self._db[uid]
            # Limpiar mapping track->person de esa persona
            stale_tracks = [
                tid for tid, p in self._track_to_person.items()
                if p == uid
            ]
            for tid in stale_tracks:
                del self._track_to_person[tid]
            # Borrar imagen del rostro
            face_path = os.path.join(self._faces_dir, f"{uid}.jpg")
            if os.path.isfile(face_path):
                try:
                    os.remove(face_path)
                except OSError as e:
                    logger.warning(
                        f"No se pudo borrar imagen {face_path}: {e}"
                    )
            self._save_db_now()
            logger.info(f"FaceReidentifier: persona {uid[:8]} borrada")
            return True

    def delete_all(self) -> int:
        """Borra TODAS las personas y sus imagenes.

        Returns:
            Cantidad de personas que habia.
        """
        with self._lock:
            count = len(self._db)
            self._db.clear()
            self._track_to_person.clear()
            # Borrar todas las imagenes
            if os.path.isdir(self._faces_dir):
                for fname in os.listdir(self._faces_dir):
                    if fname.endswith('.jpg'):
                        try:
                            os.remove(os.path.join(
                                self._faces_dir, fname))
                        except OSError as e:
                            logger.warning(
                                f"No se pudo borrar {fname}: {e}"
                            )
            # Resetear contadores de sesion
            self._session_new_count = 0
            self._session_reidentified_count = 0
            self._save_db_now()
            logger.info(
                f"FaceReidentifier: {count} personas borradas (delete_all)"
            )
            return count

    def delete_many(self, uids: list) -> int:
        """Borra multiples personas. Returns cantidad borrada."""
        deleted = 0
        for uid in uids:
            if self.delete_person(uid):
                deleted += 1
        return deleted

    def merge_persons(self, primary_uid: str,
                      secondary_uids: List[str]) -> Optional[Dict]:
        """Combina varias personas en una.

        El primary_uid retiene su UUID y se enriquece con los embeddings
        de los secondary_uids (multi-embedding). Las visitas se suman.
        Los demograficos: se mantienen los del primary, EXCEPTO si algun
        secondary tiene mayor demo_confidence -> se reemplazan.
        Las imagenes y registros de los secondary se borran.

        Args:
            primary_uid: persona que se conserva.
            secondary_uids: personas que se fusionan en primary.

        Returns:
            dict con info de la fusion, o None si primary no existe.
        """
        with self._lock:
            if primary_uid not in self._db:
                return None
            primary = self._db[primary_uid]
            primary_embeddings = list(_get_embeddings(primary))
            primary_visits = int(primary.get('visit_count', 1))
            primary_first = float(primary.get('first_seen',
                                              time.time()))
            primary_last = float(primary.get('last_seen', time.time()))
            primary_conf = float(primary.get('demo_confidence', 0.0))

            merged_count = 0
            for sec_uid in secondary_uids:
                if sec_uid == primary_uid or sec_uid not in self._db:
                    continue
                sec = self._db[sec_uid]
                # Agregar embeddings del secundario al pool
                for emb in _get_embeddings(sec):
                    primary_embeddings.append(emb)
                # Sumar visitas
                primary_visits += int(sec.get('visit_count', 1))
                # Extender rango temporal
                primary_first = min(
                    primary_first, float(sec.get('first_seen',
                                                 primary_first))
                )
                primary_last = max(
                    primary_last, float(sec.get('last_seen',
                                                primary_last))
                )
                # Si secundario tiene mayor confianza demo, usar sus demos
                sec_conf = float(sec.get('demo_confidence', 0.0))
                if sec_conf > primary_conf:
                    primary['gender'] = sec.get('gender',
                                                primary.get('gender'))
                    primary['age_range'] = sec.get(
                        'age_range', primary.get('age_range'))
                    primary['age_value'] = float(
                        sec.get('age_value', primary.get('age_value', 0)))
                    primary['demo_confidence'] = sec_conf
                    primary_conf = sec_conf
                # Re-asignar tracks del secundario al primary
                for tid, p in list(self._track_to_person.items()):
                    if p == sec_uid:
                        self._track_to_person[tid] = primary_uid

                # Borrar el secundario (registro + imagen)
                del self._db[sec_uid]
                sec_face = os.path.join(self._faces_dir, f"{sec_uid}.jpg")
                if os.path.isfile(sec_face):
                    try:
                        os.remove(sec_face)
                    except OSError as e:
                        logger.warning(f"No se pudo borrar {sec_face}: {e}")
                merged_count += 1

            # Reducir embeddings a max_count manteniendo diversidad
            max_emb = AnalyticsConfig.REID_MAX_EMBEDDINGS_PER_PERSON
            if len(primary_embeddings) > max_emb:
                primary_embeddings = self._diversify_embeddings(
                    primary_embeddings, max_emb
                )

            primary['embeddings'] = primary_embeddings
            primary.pop('embedding', None)  # limpiar formato legacy
            primary['embedding_count'] = primary.get(
                'embedding_count', 1) + merged_count
            primary['visit_count'] = primary_visits
            primary['first_seen'] = primary_first
            primary['last_seen'] = primary_last

            self._save_db_now()
            logger.info(
                f"MERGE: {primary_uid[:8]} absorbio {merged_count} "
                f"personas. Total embeddings: {len(primary_embeddings)}, "
                f"visitas: {primary_visits}"
            )
            return {
                'primary_uid': primary_uid,
                'merged_count': merged_count,
                'total_visits': primary_visits,
                'total_embeddings': len(primary_embeddings),
            }

    def _diversify_embeddings(self, embeddings: List[np.ndarray],
                              target_count: int) -> List[np.ndarray]:
        """Reduce una lista de embeddings a target_count manteniendo
        los MAS DIVERSOS entre si (algoritmo greedy farthest-first).

        Esto evita que tras merges sucesivos tengamos N embeddings casi
        identicos. Preservamos variedad de poses/condiciones.
        """
        if len(embeddings) <= target_count:
            return list(embeddings)
        # Empezar con el primero
        selected = [embeddings[0]]
        candidates = list(embeddings[1:])
        while len(selected) < target_count and candidates:
            # Encontrar el candidato mas lejano (menos similar) a TODOS
            # los ya seleccionados
            best_idx = 0
            best_min_sim = float('inf')
            for i, c in enumerate(candidates):
                # Maxima similitud contra los ya seleccionados
                max_sim_to_selected = max(
                    _cosine_similarity(c, s) for s in selected
                )
                # Queremos minimizar la max similitud (mas diverso)
                if max_sim_to_selected < best_min_sim:
                    best_min_sim = max_sim_to_selected
                    best_idx = i
            selected.append(candidates.pop(best_idx))
        return selected
