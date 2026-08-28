# Resumen del trabajo

> Qué se cambió respecto al repositorio original, por qué, y qué queda
> pendiente. Los otros dos documentos son guías de uso:
> `TRASPASO-WINDOWS.md` (levantar el servidor) y `TRASPASO-CLIENTE.md`
> (cómo funciona la interfaz).

---

## 1. Lo primero, porque cambia cómo se lee todo lo demás

**El repositorio que entregaron ya funcionaba con la interfaz.**

Está comprobado: se levantó el servidor desde el commit original
(`94845fa`), sin ningún cambio, se conectó el cliente Amazonasview real,
y procesó vídeo correctamente devolviéndolo anotado con los ROI.

No había una conexión rota que arreglar. **Lo que se hizo fue añadir un
layout que no existía**, y de paso documentar unas trampas latentes.

---

## 2. Punto de partida

Dos repositorios separados, y deben seguir así:

| Repo | Qué es |
|---|---|
| `Dsha41/Amazonas-IA-Prueba` | Servidor de inferencia. FastAPI + WebSocket |
| `Amazonas-Developers/Amazonasview` | Cliente de escritorio. PySide6 |

El servidor solo aceptaba un layout: `"Personal de Amazonas"`. Si el
cliente pedía `"Hummus"`, cerraba la conexión con código 1008.

El encargo era la analítica de Hummus: detectar **toma de orden** y
**entrega de plato** en `toma de orden y entrega de plato.avi`
(960×576, 11.9 fps, 87 s, cámara cenital fija "AM-CAJA1").

---

## 3. Las ramas

```
main                  el trabajo de analítica, sin tocar el pipeline
integracion-cliente   el cableado de Hummus en el servidor
```

Ambas subidas a `github.com/Dsha41/Amazonas-IA-Prueba`.

### Commits desde el original

```
a28db2f  Analitica de mostrador para el layout Hummus
5cb2d02  Cablear el layout "Hummus" en el servidor
da342e8  Ajustar la tolerancia de huecos a la cadencia real del servidor
b802d7f  Documento de traspaso para la maquina Windows
81465b7  Documento de traspaso del lado del cliente
49d7aab  Verificar el repositorio desde un clon limpio
```

---

## 4. Qué se tocó del código original

**Solo dos archivos, y casi todo son adiciones.**

| Archivo | Cambio | Qué se hizo |
|---|---|---|
| `src/app/app.py` | +71 −9 | Aceptar varios layouts, enrutar el procesador, comprobar que el modelo exista |
| `src/analityc/core/person_amazona_inference.py` | **+15 −0** | Un accesor nuevo. **Cero líneas eliminadas** |
| `.gitignore` | +4 | Ignorar `models/base/yolo*.pt` (39 MB regenerables) |
| `README.md`, `requirements.txt` | +14 / ±6 | De la sesión anterior: libgl1 y versiones CPU |

### El detalle de `app.py`

```python
# 1. Aceptar un conjunto en vez de un solo valor
SUPPORTED_INFERENCES = {SUPPORTED_INFERENCE, "Hummus"}
if type_inference not in SUPPORTED_INFERENCES:

# 2. El constructor recibe el layout
processor = _build_processor(client_id, config, type_inference)

# 3. Y para Hummus devuelve el envoltorio
if type_inference == "Hummus":
    return HummusProcessor(base, output_dirs={...})
```

### El accesor de `person_amazona_inference.py`

```python
def get_active_tracks(self, camera_id=1):
    """process_frame() restaura el estado global al salir, así que
    self.active_tracks queda vacío y los tracks reales están en
    camera_states."""
    return self.camera_states.get(camera_id, {}).get('active_tracks', {})
```

Es puramente aditivo. Documenta un detalle que si no se conoce hace que
leer los tracks devuelva un diccionario vacío **sin dar ningún error**.

---

## 5. Archivos nuevos

Nadie del pipeline original los importa. Si Hummus se descarta, se
borran y no queda rastro.

| Archivo | Qué es |
|---|---|
| `src/analityc/core/hummus_processor.py` | Envuelve `PersonAmazonas` y añade los eventos a `metadata["alerts"]` |
| `src/analityc/core/analytics/zone_event_tracker.py` | Entrega, por proximidad + cronómetro |
| `src/analityc/core/analytics/zone_dwell_tracker.py` | Orden, por permanencia en zona |
| `hummus_zones.json` | Zonas del mostrador y umbrales |
| `test_client_hummus.py` | Cliente de prueba que habla el mismo protocolo |
| `test_hummus_zones.py` | Análisis offline sobre el vídeo |
| `analyze_hummus.py`, `analyze_tracking.py` | Diagnóstico de umbrales y de tracking |
| `validar_hummus.py` | Validación con parámetros congelados |
| `render_video.py` | Vídeo anotado con los eventos |
| `TRASPASO-WINDOWS.md`, `TRASPASO-CLIENTE.md` | Guías |

---

## 6. Por qué envolver y no modificar

`PersonAmazonas` tiene ~3.000 líneas y da servicio al layout que ya está
en producción. Meterle ramas por tipo de layout significa que cada cambio
de Hummus arriesga romper "Personal de Amazonas".

`HummusProcessor` lo compone por fuera y delega con `__getattr__`, así que
`app.py` no distingue uno de otro.

**Un detalle que no es obvio:** `HummusProcessor.get_camera_processor()`
devuelve el procesador de cámara **también envuelto**. `app.py` hace
`processor.get_camera_processor(cam)` y procesa sobre el resultado; si se
devolviera el interno sin envolver, la analítica se saltaría por completo
**sin dar ningún error**.

---

## 7. Cómo se detecta cada evento, y por qué distinto

Los dos eventos tienen formas distintas en los datos:

**Entrega → proximidad + cronómetro.** Dibuja una V clara en la distancia
empleado-cliente: se acercan, ocurre, se separan. El mínimo coincide con
el traspaso.

**Orden → permanencia en zona.** En la caja el cajero y el cliente están a
distancia casi constante durante medio minuto. No hay ningún instante que
destaque, así que la proximidad no puede encontrarlo. Lo que sí es
evidente es que el cliente **permanece**.

### Hallazgos que fijaron los umbrales

- **YOLO no detecta los platos.** Probado con `yolo11m.pt` bajando la
  confianza a 0.10: cero detecciones de comida en el frame de la entrega.
  Por eso se detecta el patrón de interacción, no el objeto.
- **El mostrador impone un suelo de ~197 px** entre empleado y cliente.
  Nunca se acercan más. `distance_px = 250` significa "a un mostrador de
  distancia", no un número arbitrario.
- **El personal no es intercambiable.** Los cajeros se mueven entre y=385
  y y=506, los servidores entre y=63 y y=360; nadie cruza y≈370. Con un
  solo polígono de personal, la cajera —que está en el centro del
  encuadre— emparejaba con cualquiera y disparaba entregas inexistentes.
  De ahí `zona_servidor`.
- **Sin tolerancia a oclusiones**, un solo frame fallido reinicia el
  cronómetro. Medido: una oclusión de 0.6 s retrasaba una entrega 2 s.

### El fallo más grave que se encontró

Los umbrales se calibraron sobre **tiempo de vídeo a 11.9 fps**, pero el
servidor mide con **reloj de pared** y en CPU tarda ~1.5 s por frame.

Con `max_gap_sec` a 1.0 s, cada frame contaba como interrupción y **no se
detectaba nada, en silencio**. Enviando 1 frame por segundo de vídeo
sobre los 87 s completos, el servidor devolvía cero alertas mientras el
análisis offline sobre esos mismos frames encontraba 1 orden y 2 entregas.

Se resolvió haciendo que la tolerancia efectiva nunca baje de 2.5
intervalos observados (`_gap_efectivo()`), y añadiendo un aviso en el log
para que un fallo así deje de ser invisible. Verificado que el análisis
offline sigue dando exactamente lo mismo que antes.

---

## 8. Resultados

**Análisis offline** sobre el vídeo completo en pasada continua:

```
1 orden    ·  t= 0.0s
2 entregas ·  t=36.7s (la documentada en CONTEXTO.md)  y  t=73.7s
```

**Con el cliente real conectado**, los eventos llegan al panel "Alertas
IA" con su miniatura:

```
Entrega de plato — Servidor 100007 → cliente 100003 (1.2 s)
Toma de orden   — Cliente 100002 permaneció 7.3 s en la caja
```

**El cliente no necesitó ni un cambio**, porque `render_box.py:837`
consume las alertas de forma genérica: muestra el `event_type` que reciba.

---

## 9. Trampas encontradas

### Ningún modelo de `config.py` existe

Los seis faltan: `1080.pt`, `yolo12l.pt`, `best.pt`. El servidor funciona
**por accidente**: buscar `"Personal de Amazonas"` en `person_model_paths`
devuelve `None` (ese diccionario solo tiene la clave `"Hummus"`) y
`PersonAmazonas` cae a su `yolo11m.pt` por defecto.

**Cablear Hummus "bien" lo rompería**, porque ahí la búsqueda sí encuentra
`yolo12l.pt`. De ahí `_ruta_modelo_si_existe()`, que comprueba el disco
antes de usar la ruta.

### Un filtro de ROI que ignora `activate_roi`

`update_tracks()` (~línea 1620) filtra las detecciones por cercanía a
`self.roi_polygon` **siempre**, mande el cliente `activate_roi` o no. Por
defecto ese polígono es `DEFAULT_ROI`, pensado para otra cámara.

No se tocó porque es código de producción, pero conviene saberlo si
alguna cámara "no detecta nada".

### Los tracks no están donde parece

Ver el accesor de la sección 4.

---

## 10. Errores propios, para no repetirlos

**Trocear el vídeo en capturas separadas.** Se procesó en tres tandas y
cada una reiniciaba BoTSORT. Al empalmarlas parecía que el tracking
intercambiaba identidades y que una clienta eran tres. Era **artefacto
del método**: en una pasada continua sale una sola. Los dos "intercambios"
detectados caían exactamente en las costuras entre capturas.

**Saltar frames.** BoTSORT asume frames consecutivos.

**Confundir "funciona la fontanería" con "el evento es correcto".** Que
llegue una alerta al panel prueba el circuito, no la puntería.

**Dar por bueno un umbral elegido mirando los datos.** Se eligió 250 px
tras ver el resultado; eso es ajustar a la observación.

---

## 11. Verificado desde un clon limpio

Se clonó el repositorio desde cero:

| Paso | Resultado |
|---|---|
| `git clone` | Los 270 MB de pesos LFS bajan solos |
| `python scripts/setup_models.py` | Descarga `yolo11m.pt` (41 MB) |
| `python main.py` en ambas ramas | Arranca sin errores |
| Detectar los dos eventos | Funciona |

**No verificado:** `pip install -r requirements.txt`, porque se usó un
entorno que ya tenía las dependencias. Es el paso con más riesgo.

---

## 12. Lo que falta

1. **Probar la interfaz en Windows** conectada al servidor. En Linux se
   consiguió con sustitutos para las piezas de Windows, pero eso no
   sustituye a una prueba nativa.
2. Revalidar los umbrales con metraje nuevo.
3. El detector de órdenes **no distingue pedir de esperar en la cola**.
   Aquí acierta porque solo hubo una clienta en la caja.
4. Las zonas son de **esta cámara**; otra instalación necesita las suyas.
5. Preguntar a quien puso la prueba:
   - ¿Existe `1080.pt`, el modelo entrenado de Hummus, o hay que entrenarlo?
   - ¿El alcance incluye Hummus o solo "Personal de Amazonas"?

---

## 13. Limitaciones honestas

- **La entrega se infiere, no se ve.** Un empleado que se acerca sin
  entregar nada cuenta como falso positivo.
- **Los umbrales salen de un único vídeo de 87 segundos.**
- **La orden de t=0 empieza en el primer frame**, así que la clienta ya
  estaba antes de la grabación y no se sabe cuánto llevaba.
- Al reproducir el vídeo en bucle, cada vuelta genera identificadores
  nuevos: para el sistema es gente distinta. Con una cámara real no pasa.
