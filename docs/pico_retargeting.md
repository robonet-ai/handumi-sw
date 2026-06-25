# PICO Retargeting

Ultima modificacion: 2026-06-23 23:07:06 -05 -0500

## Modos de grabacion

`test/read_pico_cameras_motors.py` soporta tres modos PICO:

- `--pico-whole-body`: graba el stream original de 24 joints del cuerpo.
- `--pico-mandos`: graba headset, mandos y manos. Los joints de cuerpo y motion trackers quedan en cero. Este es el modo para probar inferencia de brazos sin trackers de pie.
- `--pico-object`: graba PICO Motion Trackers como objetos. Los joints de cuerpo quedan en cero, pero se guardan poses, velocidades, aceleraciones, cantidad de trackers y hashes de serial.

Con `--only-pico`, las camaras, `observation.state` y `action` se mantienen en el schema, pero se rellenan con ceros. Eso permite grabar episodios PICO sin conectar motores/camaras y luego reutilizar las mismas herramientas de LeRobot.

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

Para revisar un episodio whole-body antiguo, comparar codos grabados contra codos inferidos, y ver el esqueleto Axol usando el solver IK:

```bash
uv run python test/axol/visualize_pico_inferred_elbows_axol.py \
  --dataset-root outputs/datasets/dexumi-dataset-v2 \
  --episode 0 \
  --loop
```

El IK de Axol no usa los elbows grabados por PICO. Los elbows grabados se muestran solo como diagnostico para medir error. La trayectoria enviada al solver se calcula con:

- hombro PICO,
- wrist/hand PICO,
- longitudes estimadas del brazo,
- una preferencia de flexion del codo.

Por eso los trackers de pie no son necesarios para este caso de brazos. En whole-body los codos del SDK tambien vienen inferidos por el modelo corporal de PICO; si solo necesitamos manos/brazos, conviene inferir el codo de forma local y controlada desde hombro+wrist, y no depender de trackers de tobillo cuyo valor principal es pelvis/piernas/escala corporal.

## Visualizacion PICO + Piper

Piper usa un frame de URDF distinto de Axol. En este dataset, la separacion
entre hombros de PICO esta sobre su eje `z`, mientras que los brazos de Piper
se separan sobre el eje `y` de su URDF. Por eso el replay Piper usa `x,z,y`:
PICO `(x, z, y)` pasa a Piper `(frente, lateral, vertical)`.

Por defecto el replay comienza en el cero mecanico estandar de Piper: los seis
joints de cada brazo estan en `0`, con la postura plegada de referencia. Un
workspace frontal sigue disponible de forma explicita con
`--piper-workspace front`. La orientacion de la muneca permanece fuera de la
optimizacion (`--ori-weight 0`) hasta que exista una calibracion explicita
entre los quaternions PICO y el frame de cada gripper Piper.

El replay conserva ese primer frame sin volver a resolver IK y usa el limite
cinematico de Piper de aproximadamente `0.0346 rad/frame` a 30 Hz. Esto evita
que la optimizacion salte desde el cero mecanico a otra rama en un solo frame.

```bash
JAX_PLATFORMS=cpu uv run python test/piper/ik_piper_from_dataset.py \
  --dataset-root outputs/datasets/dexumi-dataset-v2 \
  --episode 0 \
  --revision main
```

Para comparar signos de los ejes alrededor del mapeo base:

```bash
JAX_PLATFORMS=cpu uv run python test/piper/compare_axis_maps_piper.py \
  --dataset-root outputs/datasets/dexumi-dataset-v2 \
  --episode 0 \
  --revision main
```

## Activar object tracking en PICO

Para object tracking no uses la calibracion full-body de tobillos. El flujo esperado es:

1. Emparejar los PICO Motion Trackers desde la app/configuracion de Motion Tracker del visor.
2. Poner el modo de los trackers en object/independent tracking cuando la app lo solicite.
3. Abrir XRoboToolkit en el PICO y conectar con el PC Service.
   - Con `--pico-wifi`, pon como IP del PC la IP LAN que imprime el script y puerto `63901`.
   - Con `--pico-adb`, conecta por USB/ADB y pon `127.0.0.1` como IP del PC.
4. En la pantalla de tracking de XRoboToolkit, activar envio de pose hacia el PC (`Send`).
5. Ejecutar el grabador con `--only-pico --pico-object --pico-wifi` o con `--pico-adb`.

Si `observation.pico.motion_tracker_count` queda en cero, el problema casi siempre esta antes de Python: los trackers no estan emparejados, el modo no es object tracking, XRoboToolkit no esta enviando ese stream, o el PC Service no esta conectado.

Para esta prueba conviene montar cada tracker rigido sobre el dorso de la mano o sobre la muneca, siempre con la misma orientacion. Sobre tela floja o una correa que rota, la posicion puede verse bien pero la orientacion va a contaminar la reconstruccion.
