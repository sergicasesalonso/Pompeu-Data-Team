# Arquitectura del código (Pompeu Data Team)

Todo nuestro código está en `pompeu/`. `evaluation/` y `starter/` son de la organización: se importan, nunca se editan. La métrica es siempre la de `evaluation/possession_metrics.py`.

## Flujo

```
data/ (ATD, XML) → pompeu.io.atd → Tracking → [clean (A1)] → método (config YAML) → etiquetas H/A/D
                                      ↑                                                  ↓
                      Línea B escribe ATD+ (.npz)                 validate.harness → results/leaderboard.csv
```

- **`Tracking`** (`pompeu/io/contract.py`) es el contrato entre las dos líneas. La Línea A solo lo lee y la Línea B solo lo escribe. Al crearse comprueba sus invariantes (formas, NaN coherentes, `ball_source`). Para cambiar de ATD a ATD+ basta con cambiar `tracking:` en el YAML.
- Coordenadas en metros (105,3 × 68,0), y la fila *i* es el frame de vídeo `frame0 + i`.

## Módulos

| Ruta | Qué hace |
|---|---|
| `pompeu/io/atd.py` | Parsea el `.txt` del ATD a numpy (`P[frame, slot, (x,y,s)]`, `B[frame, (x,y,s)]`). Tarda ~5 s la primera vez; luego se lee de la caché `.npz` |
| `pompeu/io/contract.py` | `Tracking` y `load_tracking("atd" \| ruta.npz)` |
| `pompeu/io/gt.py`, `tagging.py`, `homography.py` | GT pública, Smart Tagging (equipo → 0/1/−1) y homografía en metros |
| `pompeu/clean/` | A1: limpieza del tracking (`Tracking` → `Tracking`). Ver la sección siguiente |
| `pompeu/features/` | A2: matriz de rasgos por frame. Ver la sección "Rasgos por frame (A2)" |
| `pompeu/events/deadball.py` | A3: detector de balón muerto (P(D), tramos `D`, reinicios). Ver la sección "Balón muerto (A3)" |
| `pompeu/models/baseline.py` | El baseline oficial sobre `Tracking` (lo reproduce exacto) |
| `pompeu/models/deadball_vote.py` | A3: voto del baseline dentro de los tramos vivos, con `D` del detector (método `deadball_vote`) |
| `pompeu/pipeline.py` | Registro `METHODS`: `config → etiquetas`. Si la config tiene `clean:`, limpia primero. Cada método nuevo se registra aquí |
| `pompeu/validate/harness.py` | Los 4 bancos y el leaderboard |
| `pompeu/validate/stress.py` | Borra el balón en la ventana pública (30/50/70 %, 20 semillas) con la distribución real de huecos |
| `pompeu/validate/bootstrap.py` | Intervalos 5–95 % sobre el código oficial de la métrica |
| `decode/ external/ vision/` | Vacíos, para las etapas A4–A10 y la Línea B |

## Limpieza del tracking (A1)

`clean_tracking(tracking, cfg) → (Tracking limpio, informe)` en `pompeu/clean/__init__.py`. El contrato no cambia: la salida es otro `Tracking`. Pasos, en orden:

| Paso | Fichero | Qué hace (valores por defecto) |
|---|---|---|
| Balón: spikes | `clean/ball.py` | Quita los puntos que implican > 40 m/s respecto a los dos vecinos. Repite hasta que no quede ninguno |
| Balón: fragmentos | `clean/ball.py` | Parte la traza en huecos y saltos, y quita los fragmentos de < 3 frames (incluye las ráfagas de 2 puntos que los spikes no ven) |
| Balón: huecos | `clean/ball.py` | Rellena huecos de < 0,5 s con un spline cúbico (≤ 4 puntos a cada lado), salvo que hiciera falta > 40 m/s. `ball_source = interp`, confianza 0,5 |
| Jugadores: congelados | `clean/players.py` | Un slot que se mueve < 0,2 m en 10 s mientras la mediana del resto va a > 1 m/s. Por defecto se **marca** con confianza 0,3 (`action: flag`); con `drop` se borra |
| Jugadores: duplicados | `clean/players.py` | Dos slots del mismo equipo a < 0,5 m durante > 1 s: el del tracklet más corto pierde ese tramo |
| Jugadores: stitching | `clean/stitching.py` | Une el final de un tracklet con el inicio de otro del mismo equipo si están a < 2 s y a menos de `9 m/s·Δt + 1 m`. Usa el algoritmo húngaro por componentes conexas, reempaqueta las cadenas en menos slots y rellena los huecos de ≤ 1 s en línea recta (confianza 0,5) |
| Normalización | `clean/normalize.py` | `normalized(tracking, equipo)`: rota 180° los frames en que ese equipo ataca hacia −x, para que siempre ataque hacia +x. Se aplica a demanda, no dentro de `clean_tracking` |

- **Slots 24 y 46: son los porteros**, no objetos fijos como suponía la guía. Están en x ≈ 11 m o 94 m según la parte y presentes el 90 % del tiempo. Lo que pasa es que el tracker los congela cuando salen de plano. Por eso la regla es general y por defecto no borra nada.
- **Velocidades**: no se guardan en `Tracking`. Se recalculan con `features.kinematics.ball_kinematics` y `player_kinematics`, porque la columna de velocidad del ATD viene vacía en el 89 % de los frames.
- **Configuración**: en el YAML, `pipeline.clean: {}` aplica los valores por defecto. Cada paso se puede ajustar o apagar, por ejemplo `clean: {frozen: {action: drop}, stitching: {enabled: false}}`.
- **Orden respecto al stress**: la limpieza va dentro del pipeline, así que `public_stress` borra el balón del tracking crudo y el método lo limpia igual que haría en el partido real.
- **Informe**: `python scripts/cleaning_report.py` escribe `results/cleaning_report.md` (sí se sube a git: solo contiene recuentos).
- **Resultado** (`configs/baseline_smooth50_clean.yaml` frente a `baseline_smooth50.yaml`): el baseline **iguala** (`public_clean` es idéntico y `public_stress` cambia ±0,0005). Se eliminan 1.911 de los 2.043 saltos, pero el voto mayoritario ya los absorbía. El valor de A1 está en los rasgos y toques de A2–A4. Falta comprobarlo en `external` cuando exista A6.

## Rasgos por frame (A2)

`build_features(tracking) → DataFrame` en `pompeu/features/build.py`. Devuelve una fila por frame de vídeo, en `float32` y con unas 200 columnas. Se calcula sobre el tracking **limpio**: tarda unos 35 s el partido completo y desde la caché 0,5 s.

| Grupo | Fichero | Columnas (sufijo `_H`/`_A` por equipo) |
|---|---|---|
| Balón | `features/kinematics.py` | `ball_speed`, `ball_acc`, `ball_turn_deg`, `ball_still` (< 0,5 m/s durante > 1 s), `ball_still_s`, `ball_out` (margen 0,5 m), distancias a las líneas, al centro, a la esquina y al punto de penalti, `ball_x/y_home`, `ball_x_home_last` y temporización del hueco (`ball_gap_age_s`, `_left_s`, `_len_s`) |
| Proximidad | `features/proximity.py` | `d1/d2/d3` por equipo, `d1_diff`, `nearest_team`, `n2/n5/n10` por equipo, `near_speed`, rachas hacia atrás y hacia delante (`near_streak_s`, `near_streak_fwd_s`) |
| Control del espacio | `features/pitch_control.py` | Spearman (2018) vectorizado: `pc_ball_H` y `pc_ring_mean/min/max_H` en un anillo de 5 m. Mismo λ para los dos equipos, porque quién ataca es justo lo que queremos inferir |
| Colectivo | `features/collective.py` | En el marco de ataque de cada equipo: `cx/cy`, `cvx/cvy`, `width`, `depth`, `def_line`, `front`, `compact`, `n_recover`, `n_press`, `speed_mean`, `n_vis`. Globales: `field_speed_mean/median`, `stopped_frac`, `centroid_gap`, `line_gap`. **Existen sin balón** |
| Contexto | `features/context.py` | `half`, `minute`. Smart Tagging como voto de posesión (`st_vote`: +1 local / −1 visitante), `st_zone_team`, tiempos a la fase anterior y a la siguiente, `sp_<tipo>` y `sp_team` de las jugadas a balón parado, `goal_active` |
| Ventanas | `features/windows.py` | Media, mínimo y máximo centrados en 0,5 / 2 / 5 s de 12 rasgos clave (`<col>_w2_mean`…) |

- **Con y sin balón**: todo lo que depende del balón es NaN cuando no hay balón, y `ball_present` y la temporización del hueco lo indican. No se imputa nada, porque LightGBM y el modelo secuencial aceptan NaN y así ven que falta el balón.
- **Semántica del Smart Tagging**, comprobada con la ventana pública: los códigos de ataque (BUILD UP, PROGRESSION, CREATING CHANCES, LONG BALL, ATTACKING TRANSITION, SET PIECES) indican el equipo **con** balón. Los DEFENDING y DEFENSIVE TRANSITION indican el equipo **sin** balón. Los de zona (FINAL THIRD, IN THE BOX) no votan.
- **Sin Smart Tagging** (datos externos de A6) las columnas existen y valen NaN, así que el esquema es siempre el mismo.
- **Generar**: `python scripts/build_features.py`. Opciones: `--config X` (tracking y `clean` de esa config), `--no-clean`, `--force`. La matriz va a `cache/features_v<versión>_<hash>.parquet` (no se sube a git). El catálogo de columnas con su % de NaN va a `results/features_report.md` (sí se sube).
- **Desde código**: `from pompeu.features.build import load_or_build; F, ruta = load_or_build("atd", {})` (`{}` = limpieza por defecto, `None` = sin limpiar).
- Si cambias la definición de un rasgo, sube `FEATURES_VERSION` en `build.py` para que la caché se regenere.
- `build_features(..., groups=(...), with_windows=False)` calcula solo algunos grupos. A3 lo usa para ahorrarse el pitch control.

## Balón muerto (A3)

`detect(tracking) → DeadBall` en `pompeu/events/deadball.py`. Devuelve `p_dead` (P(D) por frame), `dead` (la segmentación más probable), `segments` y `restarts`: un DataFrame con `frame, start, type, team, confidence, source, exit`. Tarda unos 6 s el partido completo.

**Cómo decide.** Es un HMM de dos estados (vivo/muerto). Cada señal suma log-odds a favor de "muerto" o de "vivo" en cada frame. Las transiciones son pegajosas (40 s de vivo y 20 s de muerto de media), así que hace falta evidencia sostenida para cambiar de estado. Viterbi da los tramos y forward-backward da P(D). La evidencia se escala ×1/(0,5 s·fps), porque frames consecutivos no son independientes.

| Señal | Log-odds | Por qué |
|---|---|---|
| Campo lento | 3·(1,4 − velocidad media en 2 s), recortado a [−4, 3] | El umbral de 1,4 m/s de la guía |
| Balón fuera (> 0,5 m) | +8 | Muerto por reglamento. Tiene que superar el −4 de jugadores corriendo hacia el balón |
| Balón no visible | +0,5 | El tracker lo pierde más en las paradas |
| Balón quieto > 1 s en un punto de reinicio (córner, penalti, centro, banda, área pequeña) | +2 (+0,5 en otro sitio) | Balón colocado |
| Jugada a balón parado del Smart Tagging en < 3 s | +1 | |
| Fase GOALS | +2 | |
| Antes del saque inicial de cada parte | +8 | Muerto por definición. La 2ª parte se puntúa 2.934 frames antes de su saque |
| Golpeo de reinicio | −4 durante 1,5 s | El balón pasa de lento o invisible (≥ 1 s) a > 4 m/s en ≤ 0,5 s con un jugador a < 2 m |

**Después de Viterbi:**
- Se cierran los tramos vivos de < 5 s entre dos `D`: un jugador que devuelve el balón no es un reinicio.
- El inicio se adelanta al momento en que el balón cruza la línea o, si no salió, a cuando el campo empezó a frenar.
- El final se ajusta al golpeo.

**Tipo y equipo de cada reinicio:**
- Si se vio salir el balón: por banda es saque de banda para el rival del último jugador cercano. Por fondo dentro de los palos es gol y saca el que encaja. Por fondo fuera de los palos es córner si la tocó el defensor, y si no saque de puerta.
- Si no se vio salir: por dónde se coloca el balón. Si no encaja en nada, falta.
- El equipo se decide en este orden: etiqueta del Smart Tagging, luego geometría, luego el jugador que golpea. La confianza sube con las coincidencias.

**Resultados** (`configs/deadball_vote50.yaml` y `deadball_vote125.yaml` frente al baseline limpio):

| | spellF1 stress 50 % | spellF1 stress 70 % | macroF1 stress 70 % | `D` en el partido |
|---|---|---|---|---|
| baseline ±50 | 0,218 | 0,155 | 0,434 | 32 % |
| deadball_vote ±50 | 0,305 | 0,263 | 0,691 | 45,7 % |
| baseline ±125 | 0,294 | 0,232 | 0,479 | 22 % |
| deadball_vote ±125 | 0,341 | 0,289 | 0,701 | 45,7 % |

Mejora en todos los niveles de stress, pero con 17 posesiones los intervalos de spellF1 se solapan: **según la regla no se puede dar por aceptado hasta tener `external` (A6)**. En la ventana pública detecta 3 de los 4 `D` (se pierde uno de 5,5 s) y los reinicios caen a −33, −12 y +27 frames de la GT. El criterio de la guía (F1 > 0,90 y error mediano < 10 frames) solo se puede comprobar con `external`.

- **Informe**: `python scripts/deadball_report.py` escribe `results/deadball_report.md` (se sube a git) y la lista de reinicios en `cache/deadball_restarts.csv` (no se sube).
- **Parámetros**: `PARAMS` en `deadball.py`. Se pueden cambiar desde el YAML con `params: {deadball: {out_llr: 6}}`. Todos son valores físicos o de la guía, no ajustados con la GT. Se calibran en A6.
- **Pendiente para A6**: el clasificador LightGBM vivo/muerto entra por `detect(..., extra_llr=...)`. También falta la convención de la GT: en los 3 reinicios públicos, el `D` termina entre 10 y 33 frames después del golpeo.

## Bancos de validación

| Banco | Para qué |
|---|---|
| `public_clean` | Ventana pública tal cual. Solo comprueba que el pipeline es correcto; **nunca sirve para elegir** |
| `public_stress` | Ventana pública sin balón al 30/50/70 %. Confirma |
| `external` | Partidos externos degradados (A6). **Es el que decide**. Por ahora queda pendiente |
| `full_match_sanity` | Sin GT: nº de posesiones (300–400), % de `D` (35–45 %) y ninguna `X` en frames puntuados |

**Regla:** un cambio solo se acepta si mejora en `external` y en `public_stress` sin que los intervalos se solapen.

## Instalación

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.lock     # Windows (bin/ en Linux/macOS)
```

Los datos van en `data/` (o en la ruta de `MTAH_DATA_DIR`). La caché va en `cache/` (o en la ruta de `POMPEU_CACHE_DIR`).

## Ejecutar experimentos

1. Crea `configs/mi_experimento.yaml`:
   ```yaml
   name: mi_experimento
   tracking: atd              # o la ruta de un .npz de Tracking (ATD+)
   pipeline:
     method: baseline_nearest # un método registrado en pompeu/pipeline.py
     params: {window: 50}
     clean: {}                # opcional: limpieza A1 antes del método
   ```
2. Valídalo:
   ```bash
   python scripts/run_validation.py --config configs/mi_experimento.yaml
   python scripts/run_validation.py --config "configs/baseline_*.yaml"   # varios a la vez
   python scripts/run_validation.py --config configs/x.yaml --quick      # 5 semillas: para iterar, no para decidir
   python scripts/run_validation.py --config configs/x.yaml --no-save    # no escribe nada
   ```
   Cada ejecución guarda una fila en `results/leaderboard.csv` (con el hash de git y de la config), el detalle en `results/runs/` y el CSV del partido en `results/submissions/`. **Haz commit antes de lanzar**: si no, la fila queda marcada `+dirty`.
3. Genera la entrega:
   ```bash
   python scripts/make_submission.py --config configs/mi_experimento.yaml --out PompeuDataTeam.csv
   python evaluation/validate_submission.py PompeuDataTeam.csv
   ```
   El script imprime el `sha256` para comprobar que el CSV se reproduce idéntico.

## Tests

```bash
python -m pytest                       # todo (~2 min)
python -m pytest tests/test_io.py      # 0.2: carga del ATD (199.319 frames, 46 slots, balón 56,9 %)
python -m pytest tests/test_contract.py  # 0.3: invariantes de Tracking
python -m pytest tests/test_baseline.py  # reproduce los 4 baselines oficiales byte a byte
python -m pytest tests/test_validate.py  # 0.4: stress, bootstrap y arnés
python -m pytest tests/test_clean.py     # A1: cada paso de limpieza en casos sintéticos y en el partido
python -m pytest tests/test_features.py  # A2: balón quieto, pase, salida por banda, pitch control, colectivo, tagging; partido < 5 min
python -m pytest tests/test_deadball.py  # A3: HMM contra fuerza bruta, golpeo, saque de banda, córner/saque de puerta, partido
```

`tests/synthetic.py` construye `Tracking` sintéticos para los tests.

Los tests que necesitan `data/` se saltan solos si no están los datos.

## Reglas del repositorio

- Nunca se sube `data/` ni nada derivado (`cache/`, `*.npz`, `*.parquet`, `results/*` salvo `leaderboard.csv` y los informes `*_report.md`). El repositorio será público.
- Dependencias: se edita `requirements.in` y se regenera `requirements.lock` con `pip freeze` en un venv limpio.
- Contenedor: `container/pompeu.def` (Apptainer) instala `requirements.lock`.
