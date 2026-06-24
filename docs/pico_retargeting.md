# PICO Retargeting

Ultima modificacion: 2026-06-24 18:23:18 -05 -0500

## Modos de grabacion

`test/read_pico_cameras_motors.py` soporta tres modos PICO:

- `--pico-whole-body`: graba el stream original de 24 joints del cuerpo.
- `--pico-mandos`: graba headset, mandos y manos. Los joints de cuerpo y motion
  trackers quedan en cero. Este es el modo para probar inferencia de brazos sin
  trackers de pie.
- `--pico-object`: graba PICO Motion Trackers como objetos. Los joints de cuerpo
  quedan en cero, pero se guardan poses, velocidades, aceleraciones, cantidad de
  trackers y hashes de serial.

Con `--only-pico`, las camaras, `observation.state` y `action` se mantienen en
el schema, pero se rellenan con ceros. Eso permite grabar episodios PICO sin
conectar motores/camaras y luego reutilizar las mismas herramientas de LeRobot.

Ejemplo para grabar solo object tracking:

```bash
uv run python test/read_pico_cameras_motors.py \
  --only-pico \
  --pico-object \
  --pico-wifi \
  --repo-id local/pico_object_test \
  --output-dir outputs/datasets/pico_object_test \
  --task "pico object tracking wrists" \
  --num-episodes 1 \
  --episode-time-s 30 \
  --fps 30 \
  --start-button A
```

Ejemplo para grabar solo mandos:

```bash
uv run python test/read_pico_cameras_motors.py \
  --only-pico \
  --pico-mandos \
  --pico-wifi \
  --repo-id local/pico_mandos_test \
  --output-dir outputs/datasets/pico_mandos_test \
  --task "pico controller arm inference" \
  --num-episodes 1 \
  --episode-time-s 30 \
  --fps 30
```

## Visualizacion PICO + Axol

Para revisar un episodio whole-body antiguo, comparar codos grabados contra
codos inferidos, y ver el esqueleto Axol usando el solver IK:

```bash
uv run python test/axol/visualize_pico_inferred_elbows_axol.py \
  --dataset-root outputs/datasets/dexumi-dataset-v2 \
  --episode 0 \
  --loop
```

El IK de Axol no usa los elbows grabados por PICO. Los elbows grabados se
muestran solo como diagnostico para medir error. La trayectoria enviada al
solver se calcula con:

- hombro PICO,
- wrist/hand PICO,
- longitudes estimadas del brazo,
- una preferencia de flexion del codo.

Por eso los trackers de pie no son necesarios para este caso de brazos. En
whole-body los codos del SDK tambien vienen inferidos por el modelo corporal de
PICO. Si solo necesitamos manos/brazos, conviene inferir el codo de forma local
y controlada desde hombro+wrist, y no depender de trackers de tobillo cuyo valor
principal es pelvis/piernas/escala corporal.

## Mandos con visor montado en pecho/cuello

Cuando el visor no esta en la cabeza, `observation.pico.headset_pose` no debe
tratarse como cabeza. En nuestras pruebas el visor esta a la altura del pecho,
asi que se usa como `chest anchor`. Los mandos se convierten primero al frame
local del visor y recien ahi se reconstruyen hombros, codos y munecas.

Esto es importante: sumar un offset vertical para "poner el visor en la cabeza"
no basta si el visor esta rotado en el pecho. Primero hay que usar la orientacion
del visor para definir ejes locales:

- `x`: lateral del pecho,
- `y`: arriba local,
- `z`: profundidad local.

El visualizador experimental principal para este caso es:

```bash
uv run python test/axol/visualize_pico_controllers_upper_axol.py \
  --dataset-root outputs/datasets/pico_inputs_3poses_10s \
  --episode 0 \
  --loop
```

Por defecto usa `--body-frame headset-local`. Si se quiere depurar el error
viejo, se puede comparar contra mundo directo:

```bash
uv run python test/axol/visualize_pico_controllers_upper_axol.py \
  --dataset-root outputs/datasets/pico_inputs_3poses_10s \
  --episode 0 \
  --body-frame world \
  --loop
```

El visualizador muestra:

- esqueleto humano inferido desde visor-pecho + mandos,
- esqueleto Axol resuelto por IK,
- trayectoria de manos/end-effectors,
- trayectoria de codos humanos inferidos,
- trayectoria de codos Axol.

### Comparar metodos de reconstruccion del upper body

Para evaluar el retargeting antes de meter Axol en la comparacion:

```bash
uv run python test/visualize_pico_controller_body_methods.py \
  --dataset-root outputs/datasets/pico_inputs_3poses_10s \
  --episode 0 \
  --loop
```

Ese script compara cuatro metodos lado a lado:

- `A world naive`: usa posiciones globales directamente. Sirve como baseline
  malo; falla cuando el visor esta rotado en el pecho.
- `B chest local`: usa el frame local del visor-pecho y antropometria simple.
- `C old head offset`: usa offsets medidos de un dataset antiguo con full-body
  PICO de 24 puntos. En ese dataset se midio aproximadamente:
  `head - chest = [-0.023, 0.264, 0.001] m`,
  `neck - chest = [0.036, 0.199, 0.000] m`,
  `L shoulder - chest = [0.036, 0.103, 0.182] m`,
  `R shoulder - chest = [0.055, 0.118, -0.175] m`.
- `D first-pose 90`: usa la primera pose del episodio como calibracion de
  brazos flexionados a 90 grados.

Estos metodos todavia son experimentales. Con solo dos mandos y un visor en
pecho, el codo no es observable de forma unica: distintas posiciones de codo
pueden producir la misma posicion de muneca. Por eso el codo reconstruido debe
entenderse como una preferencia cinematica para ayudar al IK, no como una
medicion real.

Para mejorar este retargeting antes de usarlo en robot real, grabar al inicio
1-2 segundos quieto en una pose de calibracion conocida. La pose mas util para
estos scripts es: visor fijo en pecho/cuello, mandos al frente del pecho y codos
flexionados cerca de 90 grados. Esa pose permite estimar un pole vector de codo
mas consistente que un offset fijo.

### Visualizar datos crudos PICO

Para inspeccionar solo las trayectorias guardadas, sin reconstruccion corporal
ni IK:

```bash
uv run python test/visualize_pico_input_trajectories.py \
  --dataset-root outputs/datasets/pico_inputs_3poses_10s \
  --episode 0 \
  --ee-source both \
  --loop
```

Usar `--ee-source controllers` si solo se quieren mandos, `--ee-source trackers`
si solo se quieren object trackers, o `--ee-source both` para comparar ambos. El
HMD/visor se muestra como referencia blanca.

## Activar object tracking en PICO

Para object tracking no uses la calibracion full-body de tobillos. El flujo
esperado es:

1. Emparejar los PICO Motion Trackers desde la app/configuracion de Motion
   Tracker del visor.
2. Poner el modo de los trackers en object/independent tracking cuando la app lo
   solicite.
3. Abrir XRoboToolkit en el PICO y conectar con el PC Service.
   - Con `--pico-wifi`, pon como IP del PC la IP LAN que imprime el script y
     puerto `63901`.
   - Con `--pico-adb`, conecta por USB/ADB y pon `127.0.0.1` como IP del PC.
4. En la pantalla de tracking de XRoboToolkit, activar envio de pose hacia el PC (`Send`).
5. Ejecutar el grabador con `--only-pico --pico-object --pico-wifi` o con `--pico-adb`.

Si `observation.pico.motion_tracker_count` queda en cero, el problema casi
siempre esta antes de Python: los trackers no estan emparejados, el modo no es
object tracking, XRoboToolkit no esta enviando ese stream, o el PC Service no
esta conectado.

Para esta prueba conviene montar cada tracker rigido sobre el dorso de la mano
o sobre la muneca, siempre con la misma orientacion. Sobre tela floja o una
correa que rota, la posicion puede verse bien pero la orientacion va a contaminar
la reconstruccion.
