# Traspaso: el cliente Amazonasview

> Documento sobre el lado del cliente. El del servidor es
> `TRASPASO-WINDOWS.md`; empieza por ese si lo que quieres es levantar
> los dos y verlos hablar.
>
> Aquí está lo que hay que saber de la interfaz para integrarla: cómo
> pide la inferencia, cómo recibe los resultados, y dónde engancharse.

Repo: `github.com/Amazonas-Developers/Amazonasview`

---

## 1. La conclusión primero

**El cliente no necesita cambios para que funcione un layout nuevo.**

Está verificado de punta a punta: se añadió el layout "Hummus" en el
servidor y la interfaz lo mostró en su panel de alertas sin tocar una
sola línea de su código.

La razón es que el cliente ya está bien construido para esto:

- Manda el nombre del layout en la ruta del websocket
- Pinta las alertas de forma genérica, sin tipos de evento escritos a mano
- Ignora en silencio los campos que no conoce

Si algo hay que cambiar, es en el servidor.

---

## 2. Cómo pide la inferencia

El desplegable está en `src/gui/components/custom_status_bar.py:80`:

```python
self.layout_selector.addItems(['Seleccione...', 'Hummus', 'HummusVLM',
    'Autolavado', 'Perimetrales', 'PerimetralesMultiCam',
    'Personal de Amazonas', 'Misters'])
```

Al elegir uno, `windows_main.py:164` arma la conexión:

```python
def socket_init(self, parameter):
    self.socket.url            = "ws://127.0.0.1:9000/ws"
    self.socket.type_inference = parameter
    self.socket.conect_server()
```

Y `core/network/socket_client.py:50` los junta:

```python
self.client.open(QUrl(f'{self.url}/{self.type_inference}'))
```

O sea: elegir "Hummus" abre `ws://127.0.0.1:9000/ws/Hummus`.

**El servidor lo recibe como parámetro de ruta** (`@app.websocket("/ws/{type_inference}")`),
así que el mecanismo de layouts ya está completo en ambos lados. Solo hay
que asegurarse de que el servidor acepte el nombre.

---

## 3. Cómo recibe los resultados

`src/gui/components/render_box/render_box.py:837`:

```python
list_alert = metadata.get("alerts", []) or []
for iteration in list_alert:
    event_type = iteration.get("event_type", "Alerta")   # lo que mande el servidor
    self.alert_received.emit({
        "event_type":   event_type,
        "class_name":   iteration.get("class_name", event_type),
        "description":  iteration.get("description", ""),
        "timestamp":    iteration.get("timestamp", ""),
        "image_base64": img_b64,
        ...
    })
```

**No hay ningún tipo de evento escrito a mano.** Muestra el texto que
reciba. Por eso "Toma de orden" y "Entrega de plato" aparecieron sin
tocar nada.

### El contrato

Para que un evento salga en el panel lateral, el servidor debe meterlo en
`metadata["alerts"]` con esta forma:

```python
{
    "event_type":      "Entrega de plato",     # título en el panel
    "class_name":      "entrega_de_plato",
    "description":     "Servidor 18 → cliente 3 (1.2 s)",
    "timestamp":       "2026-08-27 21:54:32",
    "image_base64":    "<jpeg en base64>",     # o "crop_image"
    "screenshot_path": "output/...jpg",
}
```

Los campos que el cliente realmente lee son los cinco primeros. Cualquier
otro se ignora sin error, así que se pueden añadir extras para depurar.

---

## 4. De dónde saca los frames

Dos caminos, y conviene distinguirlos:

### Captura de ventana (el habitual, solo Windows)

`src/workers/capture_woker.py` usa `win32gui` para copiar el contenido de
la ventana del visor del DVR. Se lanza como **subproceso**, no se importa,
así que en Linux no rompe los imports — simplemente no está disponible.

### RTSP (funciona en cualquier sistema)

`src/workers/rtsp_worker.py`:

```python
cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
```

Se arranca desde `render_box.start_dvr_stream(channel_data)`, que lee
`channel_data["rtsp_main"]`.

**`cv2.VideoCapture` acepta la ruta de un archivo igual que una URL
`rtsp://`.** Eso permite alimentar la interfaz con un vídeo local sin
tocar su código:

```python
caja.start_dvr_stream({
    "rtsp_main":    "/ruta/al/video.avi",
    "channel_id":   "prueba",
    "channel_name": "Mostrador",
    "device_alias": "Vídeo local",
})
```

Cuando el archivo termina, `RTSPWorker` reabre la captura tras
`reconnect_delay`, así que el vídeo se repite en bucle.

Después hay que activar el envío al servidor con
`caja.activate_modesmart()` (el botón de IA en la barra de la caja).

---

## 5. El único cambio que el cliente podría necesitar

La dirección del servidor está fija en `windows_main.py:164`. Para
apuntar a otra máquina hace falta una línea, y el propio cliente ya usa
variables de entorno en otros sitios (`jarvis_url`, `name_project`), así
que sigue su convención:

```python
self.socket.url = os.getenv("AMAZONAS_WS_URL", "ws://127.0.0.1:9000/ws")
```

Compatible hacia atrás. Con el servidor en la misma máquina ni siquiera
hace falta.

Si el servidor está en un Codespace, hay que usar la URL pública con
`wss://` y marcar el puerto 9000 como público.

---

## 6. Desajustes con el servidor (inofensivos)

Los dos repositorios están algo desfasados, pero no importa: el servidor
lee todo con `data.get(campo, defecto)`.

| Campo | Cliente | Servidor |
|---|---|---|
| `image`, `roi_coordinates`, `roi_activate` | envía | lee |
| `entrega_roi_*`, `camera_id`, `cosmetics_enabled` | envía | lee |
| `door_roi_*`, `door_direction*`, `track_classes` | envía | **no existen** — se ignoran |
| `prueba1_roi_*`, `prueba2_roi_*` | **no envía** | los espera — caen a su valor por defecto |

No hay que arreglar nada de esto.

---

## 7. Correrlo en Linux (solo si hace falta)

En Windows va nativo y esto no se necesita. Pero se probó en un Codespace
y funcionó, por si sirve para CI o para una demo sin Windows.

La dependencia de Windows está **acotada a la captura de ventana**. Se
cubre con sustitutos vacíos, sin tocar el código del cliente:

| Módulo | Sustituto |
|---|---|
| `win32gui`, `win32ui`, `win32con`, `win32api` | Devuelven vacío |
| `pythoncom` | No hace nada |
| `pyautogui` | Registra en el log |
| `ctypes.WINFUNCTYPE` | Se parchea a `CFUNCTYPE` antes de importar |

Además hacen falta: `xvfb`, las librerías Qt del sistema (`libegl1`,
`libxkbcommon-x11-0`, los `libxcb-*`), y `PySide6`.

Con eso arranca bajo `xvfb-run`, y con `x11vnc` + `noVNC` se puede
manejar desde el navegador.

**Los frames hay que inyectarlos por el camino RTSP** (sección 4), porque
la captura de ventana es justo lo que no funciona.

---

## 8. Lo que no se ha probado

- Una cámara real por RTSP (solo archivos locales)
- El SDK de Hikvision/Dahua
- La captura de ventana del DVR en Windows con el layout Hummus
- Varias cámaras a la vez (hay 16 `render_box`; solo se usó uno)

---

## 9. Si vas a modificar el cliente

Piénsalo dos veces. Hasta ahora **no ha hecho falta ni un cambio**, y eso
es un buen argumento para la entrega: la integración encajó en los
mecanismos que el cliente ya tenía.

Si aun así hace falta, lo más probable es que sea:

1. La URL configurable (sección 5) — cambio pequeño y claramente correcto
2. Añadir un layout al desplegable, si se inventa uno nuevo
3. Un panel específico si las alertas genéricas no bastan

Cualquier otra cosa merece preguntarse antes si el problema no se
resuelve mejor en el servidor.
