# Traspaso: conectar el cliente en Windows

> Documento para una sesión nueva de Claude, en la máquina Windows.
> Contiene lo verificado, lo que falta y los errores que ya cometimos
> para no repetirlos.

---

## 0. Cómo quiero que trabajes conmigo

**Explícame antes de aplicar.** Esto es una prueba técnica para un empleo.
Si resuelves algo sin que yo entienda el porqué, quedo mal cuando me
pregunten. Antes de modificar código, dime qué vas a cambiar y por qué.

**No asumas: lee el código.** Casi todo lo de este documento se descubrió
leyendo la fuente y midiendo, no adivinando.

**Corrígeme si me equivoco.** Y corrígete a ti mismo: en la sesión
anterior hubo dos conclusiones que resultaron falsas y hubo que
desdecirse (están señaladas abajo).

**Sé explícito con las limitaciones.** Si algo es un apaño o tiene un caso
en que falla, quiero saberlo para poder explicarlo en la entrega.

---

## 1. Qué hay

| Repo | Qué es | Dónde |
|---|---|---|
| Servidor | FastAPI + WebSocket, inferencia | `github.com/Dsha41/Amazonas-IA-Prueba` |
| Cliente | PySide6, escritorio | `github.com/Amazonas-Developers/Amazonasview` |

**Son dos proyectos separados y deben seguir siéndolo.** No los juntes en
un repositorio.

Ramas del servidor:

- `main` — el trabajo de analítica, sin tocar el pipeline
- `integracion-cliente` — **aquí está el cableado de Hummus**

---

## 2. EMPIEZA POR AQUÍ: comprobar que lo básico ya funciona

Esto es lo primero y lo más importante. **El repositorio tal como lo
entregaron YA funciona con la interfaz.** Está verificado: se levantó el
servidor desde el commit original (`94845fa`), sin ningún cambio, se
conectó el cliente real, y procesó vídeo correctamente.

Así que no hay que "arreglar" la conexión. Confírmalo antes de tocar nada:

### 2.1 Levantar el servidor

```bash
git clone https://github.com/Dsha41/Amazonas-IA-Prueba.git
cd Amazonas-IA-Prueba
git lfs install && git lfs pull          # pesos .onnx/.caffemodel
pip install -r requirements.txt
python scripts/setup_models.py           # descarga yolo11m.pt (39 MB)
python main.py                           # escucha en :9000
```

Comprueba que responde:

```bash
curl http://localhost:9000/
```

En `main` debe decir `"supported_inference": "Personal de Amazonas"`.

> **Nota sobre `requirements.txt`:** está editado para CPU (sin `+cu118`,
> `onnxruntime` en vez de `onnxruntime-gpu`). En Windows con GPU NVIDIA
> conviene revertirlo a las versiones CUDA. El propio archivo lo documenta.

> **`libgl1` NO hace falta en Windows.** Ese requisito del README es solo
> para Linux headless.

> **El puerto se cambia con `--port`, no con la variable de entorno.**
> `config.py` declara `SERVER_PORT` pero `main.py` no la lee:
> ```bash
> python main.py --port 9100        # asi si
> SERVER_PORT=9100 python main.py   # se ignora
> ```

### Verificado desde un clon limpio

Se clono el repositorio desde cero y se comprobo lo siguiente:

| Paso | Resultado |
|---|---|
| `git clone` | Los 270 MB de pesos LFS bajan solos |
| `hummus_zones.json` y el video | Incluidos |
| `python scripts/setup_models.py` | Descarga `yolo11m.pt` (41 MB) |
| `python main.py` en ambas ramas | Arranca sin errores |
| Detectar los dos eventos | Funciona |

**Lo unico NO verificado es `pip install -r requirements.txt`**, porque se
uso un entorno que ya tenia las dependencias. Ese es el paso con mas
riesgo (versiones de Python, ruedas de PyTorch) y conviene probarlo.

### 2.2 Levantar el cliente

```bash
git clone https://github.com/Amazonas-Developers/Amazonasview.git
cd Amazonasview
pip install -r requirements.txt
python src/main.py
```

En Windows funciona nativo: `pywin32`, la captura de ventana del DVR y
todo lo demás. No hace falta ningún apaño.

### 2.3 Conectar los dos

En la interfaz, abajo a la derecha hay un desplegable **"Tipos de
inferencias"**. Elige **"Personal de Amazonas"**.

Debe aparecer **"Server connect"** en verde. Eso es todo: el cliente ya
manda el nombre del layout en la ruta del websocket
(`ws://127.0.0.1:9000/ws/Personal de Amazonas`) y el servidor ya lo lee.

**Si esto funciona, la integración base está bien y no hay que cambiar
nada de ella.**

### 2.4 Meterle imágenes

El cliente obtiene los frames capturando la ventana del visor del DVR.
En Windows eso sí funciona. Alternativas si no tienes un DVR a mano:

- Pestaña **Dispositivos** → añadir un DVR por IP/RTSP
- O abrir cualquier reproductor de vídeo y capturar su ventana

Luego pulsa el botón de **modo IA** en la caja de vídeo para que empiece
a enviar al servidor.

---

## 3. Lo que añadimos (rama `integracion-cliente`)

Solo después de confirmar el punto 2.

```bash
git checkout integracion-cliente
python main.py
curl http://localhost:9000/     # ahora lista Hummus y Personal de Amazonas
```

En la interfaz, elige **"Hummus"** en el desplegable. Antes cerraba la
conexión con código 1008; ahora conecta.

### Qué detecta

| Evento | Cómo | Por qué así |
|---|---|---|
| Entrega de plato | Proximidad empleado-cliente + cronómetro | La entrega dibuja una V clara en la distancia: se acercan, ocurre, se separan |
| Toma de orden | Permanencia del cliente en la zona de caja | En la caja ambos están a distancia casi constante medio minuto; no hay instante que destaque, así que la proximidad no sirve |

Los eventos salen por `metadata["alerts"]`, el canal que el servidor ya
usaba para las alertas de pickup. **El cliente los pinta sin ningún
cambio** porque `render_box.py:837` muestra el `event_type` que reciba.

### Archivos

| Archivo | Qué es |
|---|---|
| `src/analityc/core/hummus_processor.py` | Envuelve `PersonAmazonas` sin modificarlo |
| `src/analityc/core/analytics/zone_event_tracker.py` | Entrega por proximidad |
| `src/analityc/core/analytics/zone_dwell_tracker.py` | Orden por permanencia |
| `hummus_zones.json` | Las zonas del mostrador y los umbrales |
| `test_client_hummus.py` | Cliente de prueba que habla el mismo protocolo |
| `test_hummus_zones.py` | Análisis offline sobre el vídeo |
| `render_video.py` | Genera un vídeo anotado con los eventos |

Cambios en archivos existentes: **4 líneas en `app.py`** y **un método
nuevo de 4 líneas** en `person_amazona_inference.py`. Nada más.

### Probar sin la interfaz

```bash
python test_client_hummus.py --layout Hummus --desde 0 --frames 86 --salto 12
python test_client_hummus.py --layout "Personal de Amazonas" --frames 5   # regresión
```

---

## 4. Hechos verificados (no los vuelvas a deducir)

- **El protocolo es tolerante.** El servidor lee todo con
  `data.get(campo, defecto)`. Los campos que el cliente manda de más
  (`door_roi_*`, `track_classes`) se ignoran; los que no manda
  (`prueba1_roi_*`) caen a su valor por defecto.
- **La URL del cliente está fija** en `windows_main.py:164`. Para apuntar
  a otra máquina, una línea, siguiendo la convención de variables de
  entorno que el propio cliente ya usa en otros sitios:
  ```python
  self.socket.url = os.getenv("AMAZONAS_WS_URL", "ws://127.0.0.1:9000/ws")
  ```
- **El vídeo del mostrador**: `toma de orden y entrega de plato.avi`,
  960×576, 11.9 fps, 87 s. Cámara fija "AM-CAJA1".
- **En ese vídeo ocurre 1 toma de orden y 2 entregas.** El análisis
  offline las encuentra las tres.
- **YOLO no detecta los platos.** Probado con `yolo11m.pt` bajando la
  confianza a 0.10: cero detecciones de comida en el frame de la entrega.
  Por eso se detecta el patrón de interacción, no el objeto.
- **El mostrador impone un suelo de ~197 px** entre empleado y cliente.
  Nunca se acercan más. Por eso `distance_px` es 250: significa "a un
  mostrador de distancia".
- **El personal no es intercambiable.** Los cajeros se mueven entre
  y=385 y y=506; los servidores entre y=63 y y=360. Nadie cruza y≈370.
  De ahí `zona_servidor`, que evita que la cajera dispare entregas.

---

## 5. Trampas

### 5.1 Ningún modelo del `config.py` existe

Los seis faltan: `1080.pt`, `yolo12l.pt`, `best.pt`. El servidor funciona
**por accidente**: buscar `"Personal de Amazonas"` en `person_model_paths`
devuelve `None` (ese diccionario solo tiene la clave `"Hummus"`), y
`PersonAmazonas` cae a su `yolo11m.pt` por defecto.

**Cablear Hummus "bien" lo rompería**, porque ahí la búsqueda SÍ encuentra
`yolo12l.pt`, que no existe. Por eso `_ruta_modelo_si_existe()` en
`app.py` comprueba el disco antes de usar la ruta.

Al arrancar verás `WARNING: Modelo ... no existe en disco`. Es esperado.

### 5.2 Los tracks no están donde parece

`process_frame()` intercambia el estado por cámara al entrar y lo
restaura al salir. Cuando retorna, `self.active_tracks` está **vacío** y
los tracks reales están en `camera_states[camera_id]['active_tracks']`.

Leer `active_tracks` directamente devuelve un diccionario vacío **sin dar
ningún error**. Usa `get_active_tracks(camera_id)`.

### 5.3 Un filtro de ROI que ignora `activate_roi`

`update_tracks()` (~línea 1620 de `person_amazona_inference.py`) filtra
las detecciones por cercanía a `self.roi_polygon` **siempre**, mande el
cliente `activate_roi` o no. Por defecto ese polígono es `DEFAULT_ROI`,
pensado para otra cámara.

No se tocó porque es código de producción, pero conviene saberlo si
alguna cámara "no detecta nada".

### 5.4 El reloj del servidor no es el del vídeo

Los umbrales se calibraron sobre tiempo de vídeo a 11.9 fps, pero el
servidor mide con reloj de pared. En CPU tarda ~1.5 s por frame.

Con `max_gap_sec` a 1.0 s, **cada frame contaba como interrupción y no se
detectaba nada, en silencio**. Se resolvió haciendo que la tolerancia
efectiva nunca baje de 2.5 intervalos observados (ver `_gap_efectivo()`).

En Windows con GPU el servidor irá mucho más rápido y esto dejará de
importar, pero merece una comprobación.

---

## 6. Limitaciones honestas

- **El detector de órdenes no distingue pedir de esperar en la cola.** En
  este vídeo acierta porque solo hubo una clienta en la caja.
- **Las zonas son de esta cámara.** Otra instalación necesita las suyas.
- **Los umbrales se calibraron sobre un único vídeo de 87 segundos.**
- **La entrega se infiere, no se ve.** Un empleado que se acerca sin
  entregar nada cuenta como falso positivo.
- **La orden de t=0 empieza en el primer frame**, así que la clienta ya
  estaba antes de que arrancara la grabación y no se sabe cuánto llevaba.

---

## 7. Errores que ya cometimos (no repetir)

**Trocear el vídeo en capturas separadas.** Se procesó en tres tandas y
cada una reiniciaba BoTSORT. Al empalmarlas parecía que el tracking
intercambiaba identidades y que una clienta eran tres. **Era artefacto
del método**: en una pasada continua sale una sola. El servidor procesa
un flujo continuo, así que el problema no existe en producción.

**Saltar frames.** BoTSORT asume frames consecutivos. Con salto, los
track_id se reasignan y ningún par sobrevive.

**Confundir "funciona la fontanería" con "el evento es correcto".** Que
llegue una alerta al panel prueba el circuito, no la puntería. Para
juzgar si acierta hay que mirar el análisis offline, donde el tiempo es
el del vídeo.

**Dar por bueno un umbral elegido mirando los datos.** Se eligió 250 px
tras ver el resultado; eso es ajustar a la observación. Solo vale si se
prueba en metraje distinto sin retocarlo.

---

## 8. Lo que falta

1. **Confirmar el punto 2** en Windows (lo básico, sin Hummus)
2. Probar Hummus con la interfaz nativa
3. Revalidar los umbrales con vídeo nuevo
4. Decidir si el detector de órdenes necesita otra señal
5. Preguntar a quien puso la prueba:
   - ¿Existe `1080.pt`, el modelo entrenado de Hummus, o se espera que se entrene?
   - ¿El alcance incluye Hummus o solo "Personal de Amazonas"?

---

## 9. Qué se probó en Linux y qué no

Se levantó todo en un Codespace, incluida la interfaz gráfica con Xvfb y
sustitutos para las piezas de Windows. Funcionó de punta a punta: el
cliente conectó, envió vídeo y mostró las dos alertas en su panel.

**Pero eso fue con apaños.** En Windows debería ir mejor y más limpio.
Lo que allí no se pudo probar:

- La captura de ventana del DVR (es lo que se sustituyó)
- Una cámara real por RTSP
- El rendimiento con GPU
