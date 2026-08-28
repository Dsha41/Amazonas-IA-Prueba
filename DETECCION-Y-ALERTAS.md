# Cómo funcionan las detecciones y las alertas

> Qué se cambió para que los eventos se detecten bien y lleguen al panel
> del cliente, con el código esencial de cada pieza.
>
> Para el contexto general ver `RESUMEN.md`.

---

## 1. Los ROI: por qué no se pueden quitar

Lo primero, porque genera confusión: **esos polígonos y recuadros que se
ven sobre el vídeo no son decoración. Son el mecanismo con el que el
sistema decide qué mira y qué cuenta.**

El cliente manda cuatro ROI en cada frame:

| Campo que envía el cliente | Llega a `process_frame` como | Para qué sirve |
|---|---|---|
| `roi_coordinates` | `roi` | **El principal.** Filtra detecciones y cuenta entradas/salidas |
| `entrega_roi_coordinates` | `roi_escaparate` | Zona verde. Hummus la usa como zona de entrega |
| `prueba1_roi_coordinates` | `roi_prueba1` | Zonas violeta. **El cliente nunca las envía** |
| `prueba2_roi_coordinates` | `roi_prueba2` | Igual |

### El ROI principal tiene dos trabajos distintos

**Trabajo 1 — filtrar qué detecciones se convierten en tracks.**
En `update_tracks()` (~línea 1620 de `person_amazona_inference.py`):

```python
roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))

for det in detections:
    center = det['center']
    distance_to_roi = cv2.pointPolygonTest(
        roi_polygon_points, (int(center[0]), int(center[1])), True)

    if distance_to_roi > -50:          # dentro, o hasta 50 px fuera
        filtered_detections.append(det)
```

Todo lo que caiga a más de 50 px fuera del polígono **se descarta antes
de llegar al tracker**. No existe para el resto del sistema.

**Trabajo 2 — contar quién entra y quién sale.**
En `process_entry_exit_logic()`:

```python
def process_entry_exit_logic(self, frame):
    roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
    ...
    is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
    previous_inside = track.get('is_inside', False)

    if is_inside and not previous_inside:
        # ENTRADA: arranca cronómetro, guarda foto
    elif not is_inside and previous_inside:
        # SALIDA: cierra la visita, registra el tiempo
```

De ahí salen los contadores que se ven sobre el vídeo:
`Personas totales: 15 | Atendidas: 0/15 | Activas: 9`.

### La consecuencia práctica

**Si mueves o encoges el ROI principal, cambias qué detecta el sistema.**
No es una guía visual que se pueda ignorar: si una persona queda fuera,
para el servidor no existe.

### Una trampa que conviene conocer

Ese filtro se aplica **siempre**, mande el cliente `roi_activate` en true
o en false. El parámetro `activate_roi` controla *otro* filtro distinto
(el de la línea 2438), no este.

Por defecto `self.roi_polygon` vale `DEFAULT_ROI`, un rectángulo pensado
para otra cámara que cubre x=500-1040. Si una instalación "no detecta
nada", mirar esto primero.

---

## 2. Qué se cambió para que las detecciones fueran correctas

Cuatro problemas reales, en el orden en que aparecieron.

### 2.1 Los platos no se pueden detectar

Probado con `yolo11m.pt` sobre el frame exacto de la entrega, bajando la
confianza hasta 0.10: **cero detecciones de comida**. Los `bowl` que
aparecen están siempre en la misma coordenada — es un pan estático de la
repisa, no una bandeja.

**Decisión:** no detectar el objeto, detectar el **patrón de interacción**
que lo implica. `person` sí es fiable: 5-10 por frame, consistente.

### 2.2 El mostrador impone un suelo de distancia

Se midió la distancia mínima entre empleado y cliente sobre 400 frames:

```
distancia más corta ......... 197 px
mediana de la más corta ..... 264 px
```

**Nunca bajan de ~197 px porque el mostrador está en medio.** Un umbral
de 180 px pedía algo físicamente imposible, y por eso no detectaba nada.

`distance_px = 250` significa literalmente *"a un mostrador de
distancia"*. No es un número elegido hasta que salieran resultados.

### 2.3 El personal no es intercambiable

Con un solo polígono de personal, **la cajera disparaba entregas que
nunca ocurrieron**: está en el centro del encuadre, así que queda cerca de
todo el mundo.

Se midió dónde trabaja cada tipo de empleado:

```
CAJEROS      y = 385-506     100004 (482,438)   100001 (448,453)
SERVIDORES   y =  63-360     100018 (557,267)   100021 (641,210)
```

Nadie cruza y≈370. El corte salió de los datos, no del ojo.

De ahí `staff_pair_polygon` en `ZoneEventTracker`:

```python
if self._is_staff(center):
    # Es empleado: nunca cuenta como cliente, aunque no pueda
    # emparejar para este evento concreto.
    if (self._pair_polygon is not None
            and not self._is_inside(center, self._pair_polygon)):
        continue
    staff[tid] = track
elif in_zone:
    clients_in_zone[tid] = track
```

El empleado fuera de esa sub-área sigue siendo empleado (nunca se cuenta
como cliente), simplemente **no puede formar pareja** para ese evento.

### 2.4 Una oclusión de 0.6 s retrasaba la entrega 2 segundos

El cronómetro se reiniciaba en cuanto la pareja fallaba **un solo frame**.
Medido sobre el vídeo:

```
34.9s   201px   acumula 0.42s     ← el punto más cercano de la escena
35.2s   224px   acumula 0.76s
35.4s   254px   SE ROMPE
35.6s   NO DETECTADO  ┐  alguien se cruzó por delante
36.2s   NO DETECTADO  ┘
36.2s   230px   vuelve a cero
37.8s   238px   CONFIRMA, casi 2 s tarde
```

Se añadió tolerancia a huecos cortos:

```python
prox = self._active_proximities.get(pair)
if prox is None or (now - prox["visto"]) > self._gap_efectivo():
    # Pareja nueva, o el hueco fue tan largo que la cercanía
    # anterior se da por terminada.
    self._active_proximities[pair] = {"inicio": now, "visto": now}
else:
    # Hueco corto (una oclusión): se mantiene el cronómetro.
    prox["visto"] = now
    elapsed = now - prox["inicio"]
    if elapsed >= self._min_time_sec and cid not in self._confirmed:
        ...
```

---

## 3. Dos detectores, porque los eventos tienen formas distintas

Esto es lo que más costó entender, y es el núcleo del diseño.

### La entrega dibuja una V

```
34.2s  258  #########################
35.2s  224  ###################### <<<
36.2s  230  ###################### <<<
37.3s  210  #################### <<<   ← mínimo = la entrega
38.3s  262  ##########################
40.3s  343  ##################################
```

Se acercan, ocurre, se separan. **Hay un instante que destaca**, así que
la proximidad lo encuentra. → `ZoneEventTracker`

### La orden es una meseta plana

```
34.2s  241  ########################
37.3s  266  ##########################
44.3s  275  ###########################
48.3s  236  #######################
52.4s  232  #######################
```

El cliente está ahí **23 segundos seguidos** oscilando entre 230 y 285 px,
sin estructura. El cajero y el cliente están a distancia casi constante
todo el rato. **Ningún instante destaca**, así que ningún umbral de
proximidad puede encontrarlo.

Lo que sí es evidente en los datos es que el cliente **permanece**.
→ `ZoneDwellTracker`

```python
visita = self._visitas.get(tid)
if visita is None or (now - visita["visto"]) > self._gap_efectivo():
    self._visitas[tid] = {"inicio": now, "visto": now}
    continue

visita["visto"] = now
permanencia = now - visita["inicio"]
if permanencia >= self._min_dwell_sec and tid not in self._confirmed:
    # evento confirmado
```

No exige empleado cerca: en un mostrador el personal **siempre** está
presente, así que su cercanía no aporta información.

---

## 4. El fallo que no daba ninguna señal

El peor de todos, porque **fallaba en silencio**.

Los umbrales se calibraron sobre **tiempo de vídeo a 11.9 fps** (0.08 s
por frame), pero el servidor mide con **reloj de pared** y en CPU tarda
**~1.5 s por frame**.

Con `max_gap_sec = 1.0 s`, entre frame y frame pasaba más tiempo del
tolerado: **cada frame contaba como interrupción**, el cronómetro volvía a
cero siempre, y no se confirmaba nada. Sin error, sin aviso.

Medido: enviando 1 frame por segundo de vídeo sobre los 87 s completos,
el servidor devolvía **cero alertas**, mientras el análisis offline sobre
esos mismos frames encontraba 1 orden y 2 entregas.

### La solución: tolerancia adaptativa

```python
def _gap_efectivo(self) -> float:
    """max_gap_sec es un parámetro TÉCNICO: cuánto puedo perder de vista
    a alguien. Su valor correcto depende de cada cuánto llegan los
    frames, y eso cambia según dónde corra:

        análisis offline, 11.9 fps  ->  0.08 s/frame
        servidor en CPU             ->  ~1.5 s/frame
    """
    if len(self._intervalos) < 3:
        return self._max_gap_sec
    ordenados = sorted(self._intervalos)
    mediana = ordenados[len(ordenados) // 2]
    return max(self._max_gap_sec, 2.5 * mediana)
```

A 11.9 fps da 0.21 s, **por debajo** del 1.0 s configurado, así que el
valor configurado manda y **el análisis offline no cambia** (verificado:
sigue dando 2 entregas y 1 orden, mismos clientes). En el servidor lento
da ~3.7 s, que es lo que hacía falta.

Y un aviso para que deje de ser invisible:

```python
if intervalo > self._gap_efectivo():
    logger.warning(
        "[%s] los frames llegan cada %.1f s, más que los %.1f s "
        "configurados en max_gap_sec. Sin compensar, el cronómetro se "
        "reiniciaría en cada frame y no se confirmaría nada...",
        self.event_name, intervalo, self._max_gap_sec)
```

> **Con GPU esto deja de importar**, porque el servidor procesará mucho
> más rápido. Pero el aviso sigue siendo útil.

---

## 5. Cómo llegan las alertas al cliente

**Sin modificar una sola línea del cliente.** El canal ya existía.

### El descubrimiento

En `person_amazona_inference.py` hay un comentario revelador:

```python
# Alertas pickup -> el cliente las pinta en "Toma de Orden"
if self._pending_pickup_alerts:
    metadata['alerts'] = list(self._pending_pickup_alerts)
    self._pending_pickup_alerts.clear()
```

Y en el cliente (`render_box.py:837`), el consumo es **completamente
genérico**:

```python
list_alert = metadata.get("alerts", []) or []
for iteration in list_alert:
    event_type = iteration.get("event_type", "Alerta")   # lo que llegue
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

Para que un evento salga en el panel lateral, basta meterlo en
`metadata["alerts"]` con esta forma:

```python
{
    "event_type":      "Entrega de plato",     # título en el panel
    "class_name":      "entrega_de_plato",
    "description":     "Servidor 18 → cliente 3 (1.2 s)",
    "timestamp":       "2026-08-27 21:54:32",
    "image_base64":    "<jpeg en base64>",
    "screenshot_path": "output/...jpg",
}
```

Los cinco primeros son los que el cliente lee. Cualquier campo extra se
ignora sin error, así que se pueden añadir para depurar.

### Dónde se inyecta

En `HummusProcessor.process_frame()`:

```python
def process_frame(self, image, *args, **kwargs):
    resultado = self.inner.process_frame(image, *args, **kwargs)
    img, metadata = resultado

    tracks = self.inner.get_active_tracks(camera_id)
    if tracks:
        nuevos = []
        nuevos += self._entrega.update(tracks)
        nuevos += self._orden.update(tracks)
        for ev in nuevos:
            metadata.setdefault("alerts", []).append(
                self._construir_alerta(ev, img))

    return img, metadata
```

---

## 6. Dos detalles que no son obvios y rompen en silencio

### Los tracks no están donde parece

`process_frame()` intercambia el estado por cámara al entrar
(`_push_state`) y lo restaura al salir (`_pop_state`, en un `finally`).
Cuando retorna, **`self.active_tracks` está vacío**.

```python
def get_active_tracks(self, camera_id=1):
    """Los tracks reales quedan en camera_states."""
    return self.camera_states.get(camera_id, {}).get('active_tracks', {})
```

Leer `self.active_tracks` directamente devuelve un diccionario vacío
**sin dar ningún error**. Este accesor es el único cambio que se hizo al
pipeline: 15 líneas, cero eliminaciones.

### El envoltorio debe envolver también a sus hijos

`app.py` hace `processor.get_camera_processor(cam)` y procesa sobre el
resultado. Si ahí se devolviera el `PersonAmazonas` interno sin envolver,
**la analítica se saltaría por completo, sin error alguno**.

```python
def get_camera_processor(self, camera_id):
    if camera_id not in self._por_camara:
        self._por_camara[camera_id] = HummusProcessor(
            self.inner.get_camera_processor(camera_id),
            zonas=self.zonas, output_dirs=self._dirs)
    return self._por_camara[camera_id]
```

De paso sale bien por diseño: **cada cámara tiene sus propios
detectores**, que es lo correcto — dos cámaras no deben compartir el
cronómetro de una entrega.

---

## 7. Ajustar la zona desde la interfaz

El cliente ya tiene un editor de polígonos y manda el resultado en cada
frame como `entrega_roi_coordinates`. `app.py` lo pasa a `process_frame`
como `roi_escaparate`. **El mecanismo entero ya existía en los dos lados
y nadie lo estaba usando para Hummus.**

```python
self._aplicar_roi_del_cliente(kwargs.get("roi_escaparate"))
```

Así el operador coloca la zona de entrega **con el ratón, sobre el vídeo
real y sin parar el servidor**, en vez de escribir coordenadas a mano en
`hummus_zones.json` — que es como se pusieron estas, y por eso son
específicas de esta cámara.

Se apaga con `HUMMUS_ROI_CLIENTE=off`.

**Limitación:** solo vale para la zona de entrega. La zona de caja y la
del servidor no tienen editor propio en el cliente, y el ROI principal
(amarillo) ya se usa para filtrar detecciones — reaprovecharlo rompería
esa función.

---

## 8. Resultados

Análisis offline sobre el vídeo completo, pasada continua:

```
1 orden    ·  t= 0.0s
2 entregas ·  t=36.7s (la documentada)  y  t=73.7s
```

Con el cliente real conectado, en el panel "Alertas IA":

```
Entrega de plato — Servidor 100007 → cliente 100003 (1.2 s)
Toma de orden   — Cliente 100002 permaneció 7.3 s en la caja
```

---

## 9. Lo que estos cambios NO resuelven

- **El detector de órdenes no distingue pedir de esperar en la cola.**
  Aquí acierta porque solo hubo una clienta en la caja.
- **La entrega se infiere, no se ve.** Un empleado que se acerca sin
  entregar nada cuenta como falso positivo.
- **Las zonas son de esta cámara.** Otra instalación necesita las suyas.
- **Los umbrales salen de un único vídeo de 87 segundos**, y se eligieron
  mirando los datos. Eso es ajustar a la observación: para sostener que
  funcionan hay que probarlos en metraje nuevo sin retocarlos.
