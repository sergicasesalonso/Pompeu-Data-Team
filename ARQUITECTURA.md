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
| `pompeu/events/touches.py` | A4: eventos de balón (toques y controles) con equipo y P(local) |
| `pompeu/models/rules.py` | A4: `rules_v0`, máquina de estados de posesión. **Primera entrega real** |
| `pompeu/decode/viterbi.py` | A5: núcleos semi-Markov en numba (Viterbi y forward-backward), genéricos |
| `pompeu/models/hsmm.py` | A5: `hsmm_v1`, modelo semi-Markov oculto de la posesión |
| `pompeu/pipeline.py` | Registro `METHODS`: `config → etiquetas`. Si la config tiene `clean:`, limpia primero. Cada método nuevo se registra aquí |
| `pompeu/validate/harness.py` | Los 4 bancos y el leaderboard |
| `pompeu/validate/stress.py` | Borra el balón en la ventana pública (30/50/70 %, 20 semillas) con la distribución real de huecos |
| `pompeu/validate/bootstrap.py` | Intervalos 5–95 % sobre el código oficial de la métrica |
| `external/ vision/` | Vacíos, para A6 y la Línea B |

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

## Toques y posesión: `rules_v0` (A4)

**Eventos** (`detect_events(tracking, F)` en `pompeu/events/touches.py`, ~1,5 s el partido):

| Evento | Cuándo | Equipo |
|---|---|---|
| `touch` | La velocidad del balón cambia > 3 m/s o su dirección > 30° (comparando 2 frames antes y después), con un jugador a < 2 m. Los candidatos a ≤ 3 frames se agrupan en uno | P(local) = 80 % logística de la diferencia de distancias + 20 % pitch control. Guarda `speed_in` y `speed_out` |
| `control` | El balón está a < 1,5 m del **mismo** jugador durante ≥ 0,6 s (conducción, o balón parado junto a alguien) | El de ese jugador |

**Máquina de estados** (`pompeu/models/rules.py`). Se aplica dentro de cada tramo vivo de A3:
- **Inicio:** la posesión empieza con el equipo del reinicio (A3) o, si no se conoce, con el del primer evento.
- **Evento del poseedor:** confirma la posesión y anula cualquier toque rival pendiente (eso era un desvío).
- **Toque rival:** queda pendiente. Un segundo toque rival en < 3 s (pases encadenados) o un control rival **confirman** el cambio.
- **Balón por el aire:** un toque con el balón a > 15 m/s antes y después (pasó volando junto al jugador) se ignora.
- **Equipo dudoso:** se ignoran los eventos con P(local) entre 0,35 y 0,65.
- **Sin balón no hay eventos**, así que la posesión se mantiene. Es la regla del reto: "si el balón está perdido o disputado, sigue con el último equipo que lo tuvo".
- **Fecha del cambio:** la GT pública fecha los cambios en el **primer toque** del nuevo equipo, no cuando ya lo controla. Por eso, una vez confirmado, el cambio se adelanta hasta el inicio de la racha en que el rival era el equipo más cercano, sin pasar del último evento del poseedor (`backdate: streak`). La alternativa `touch` lo adelanta solo hasta el toque detectado, pero detectamos pocos toques (unos 8 por minuto).

**Hallazgo:** el balón del ATD a veces es falso: aparece "quieto" en pleno juego con el jugador más cercano a 7–10 m. Como todos los eventos exigen un jugador a < 2 m, esos balones fantasma nunca provocan un cambio.

**Resultados** (`configs/rules_v0.yaml`):

| | clean | stress 30 % | stress 50 % | stress 70 % | posesiones en el partido |
|---|---|---|---|---|---|
| baseline ±50 (limpio) | 0,326 | 0,261 | 0,218 | 0,155 | 594 |
| deadball_vote ±50 | 0,411 | 0,375 | 0,305 | 0,263 | 474 |
| deadball_vote ±125 | 0,529 | 0,470 | 0,341 | 0,289 | 284 |
| **rules_v0** | 0,506 | 0,466 | 0,360 | 0,294 | 250 |

- **Frente al baseline** mejora con claridad. Frente a `deadball_vote125` queda **empatado**, dentro del ruido.
- **Punto débil: falta segmentación.** Da 250 posesiones en el partido, cuando se esperan 300–400, y con el stress al 70 % detecta 8,8 posesiones frente a 17. Cuando no hay balón no hay eventos, y se pierden las posesiones cortas que caen dentro de un hueco. Eso lo tiene que resolver A5 (HSMM con rasgos colectivos y Smart Tagging, que funcionan sin balón).
- **Entrega de seguridad:** `python scripts/make_submission.py --config configs/rules_v0.yaml` genera `results/submissions/rules_v0.csv`. Es válido según `evaluation/validate_submission.py` y reproducible: dos regeneraciones dan el mismo sha256.
- **Parámetros** en el YAML: `params: {deadball: {...}, touches: {...}, rules: {...}}`, que sobrescriben los `PARAMS` de cada módulo.

## Modelo semi-Markov: `hsmm_v1` (A5)

**Inferencia** (`pompeu/decode/viterbi.py`). Contiene dos núcleos genéricos en numba:
- `viterbi`: la mejor segmentación;
- `posteriors`: P(estado) por frame, mediante forward-backward.

Ambos trabajan sobre segmentos con duración explícita. Cuestan O(N · Dmax · S) y aceptan transiciones que dependen del tiempo. `tests/test_hsmm.py` los compara con la **enumeración exhaustiva** de todas las segmentaciones de problemas pequeños, y coinciden exactamente (Viterbi, posteriores y log Z). En el partido completo, Viterbi tarda unos 6 s por parte y los posteriores unos 14 s.

**Modelo** (`pompeu/models/hsmm.py`). Tiene **4 estados ocultos y 3 etiquetas**:

| Estado | Significado | Etiqueta |
|---|---|---|
| `HL` / `AL` | Balón vivo, lo tiene el local / el visitante | `H` / `A` |
| `HD` / `AD` | Balón muerto, reinicia el local / el visitante | `D` |

Un balón muerto también tiene equipo: el que reinicia (lo dice CHALLENGE.md). Con un único estado `D`, cualquier evidencia de equipo durante una parada (por ejemplo, el portero colocando el balón) se interpretaba como "balón vivo" y el modelo se comía las paradas. Con `HD`/`AD` la evidencia de equipo y la de vivo/muerto se **factorizan**: `log e(equipo, vivo) = ±L_equipo/2 ± L_muerto/2`. Un test comprueba que la evidencia de equipo nunca cambia P(vivo)/P(muerto). Además, el reinicio pasa a formar parte de la estructura: `HD → HL`.

- **Duraciones:** log-normales. Las posesiones tienen una mediana de 8 s (mínimo 0,2 s) y las paradas de 20 s (mínimo 1 s). Un segmento de más de 120 s se representa como dos segmentos del mismo estado.
- **Transiciones:**
  - posesión → pérdida (0,7) o parada (0,3), y en la parada reinicia cualquiera de los dos equipos al 50 %;
  - parada → posesión del mismo equipo (0,98).

**Expertos de vivo/muerto (`L_muerto`):**
- las señales de A3, con el mismo escalado que en A3;
- la segmentación final de A3, con ±3 por observación. Aporta lo que las señales sueltas no tienen: el inicio en el cruce de la línea, el final en el golpeo de reinicio y la regla de que un balón manipulado durante la parada no es un reinicio. Por eso se descartan los golpeos que A3 dejó dentro de una parada.

**Expertos de equipo (`L_equipo`):**
- **Proximidad:** cuenta solo si el balón está al alcance de alguien (< 4 m) y **no está disputado**. Si los dos equipos están a menos de 2,5 m, no aporta nada, porque un balón disputado sigue con quien lo tenía.
- **Eventos de A4:** los controles y los toques, sin contar los balones que pasan volando.
- **Smart Tagging:** voto de posesión con una correlación de 5 s, porque una etiqueta es **una** decisión por mucho que dure.
- **Reinicio de A3:** una única observación sobre el equipo de la parada.

**Resultados** (20 semillas):

| | clean | stress 30 % | stress 50 % | stress 70 % | macroF1 s50 / s70 | posesiones |
|---|---|---|---|---|---|---|
| rules_v0 | **0,506** | **0,466** | 0,360 | 0,294 | 0,734 / 0,684 | 250 |
| **hsmm_v1** | 0,449 | 0,410 | 0,352 | **0,322** | **0,767 / 0,743** | 238 |
| hsmm_v1 sin proximidad | 0,489 | 0,439 | 0,349 | 0,314 | 0,770 / 0,736 | 191 |
| hsmm_v1_rules (`rules_llr: 1`) | 0,491 | 0,451 | 0,352 | 0,292 | 0,734 / 0,685 | 244 |

- **No cumple todavía el criterio del plan** (mejorar a `rules_v0` en `public_stress`). Es mejor cuando falta mucho balón y peor cuando se ve. El partido real tiene un ~41 % de balón ausente, que cae entre stress 30 % y 50 %, y ahí queda empatado o algo peor. **La entrega sigue siendo `rules_v0`.**
- La desventaja con balón está en **dónde coloca los cambios**. `rules_v0` codifica la convención de la GT ("espera al control y adelanta el cambio al primer toque") y el HSMM tiene que deducirla de la evidencia frame a frame. Si se le pasa `rules_v0` como experto (`hsmm_v1_rules`), se limita a copiarlo.
- Lo que falta es lo que el plan ya prevé: **A6**, para ajustar los pesos en `external`, y **A7**, una emisión aprendida de quién tiene el balón cuando no se ve. El HSMM es además el motor de posteriores que necesita el MBR de A9 (`decode(..., {"posteriors": True})`).
- **Sobre el rango de 300–400 posesiones de la guía:** la GT pública tiene posesiones de 14,2 s de media. Con un 45 % de `D` salen unas **246** posesiones, y con un 35 %, unas 291. Ese rango no es coherente con el propio rango de `D` del plan. Nuestros modelos (238–250 posesiones de 14–15 s) sí lo son con la GT.
- **Configs:** `configs/hsmm_v1.yaml` y `configs/hsmm_v1_rules.yaml`. Los parámetros se cambian en `params: {hsmm: {...}, deadball: {...}, touches: {...}}`. Ninguno se ha ajustado con la GT: son valores físicos o del plan, y el único que se ha elegido con `public_stress` es el peso de la segmentación de A3 (3).
- **Entorno:** numba 0.66 está fijado en `requirements.lock`. En Windows, crea el venv en una ruta corta (por ejemplo `C:\venvs\pompeu`): si la ruta es larga, la DLL de numba supera el límite de 260 caracteres y no carga.

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
python -m pytest                       # todo (~4 min)
python -m pytest tests/test_io.py      # 0.2: carga del ATD (199.319 frames, 46 slots, balón 56,9 %)
python -m pytest tests/test_contract.py  # 0.3: invariantes de Tracking
python -m pytest tests/test_baseline.py  # reproduce los 4 baselines oficiales byte a byte
python -m pytest tests/test_validate.py  # 0.4: stress, bootstrap y arnés
python -m pytest tests/test_clean.py     # A1: cada paso de limpieza en casos sintéticos y en el partido
python -m pytest tests/test_features.py  # A2: balón quieto, pase, salida por banda, pitch control, colectivo, tagging; partido < 5 min
python -m pytest tests/test_deadball.py  # A3: HMM contra fuerza bruta, golpeo, saque de banda, córner/saque de puerta, partido
python -m pytest tests/test_rules.py     # A4: toques, controles, cada regla de la máquina de estados, intercepción, partido
python -m pytest tests/test_hsmm.py      # A5: Viterbi y posteriores contra enumeración exhaustiva, propiedades del modelo, escenas, partido
```

`tests/synthetic.py` construye `Tracking` sintéticos para los tests.

Los tests que necesitan `data/` se saltan solos si no están los datos.

## Reglas del repositorio

- Nunca se sube `data/` ni nada derivado (`cache/`, `*.npz`, `*.parquet`, `results/*` salvo `leaderboard.csv` y los informes `*_report.md`). El repositorio será público.
- Dependencias: se edita `requirements.in` y se regenera `requirements.lock` con `pip freeze` en un venv limpio.
- Contenedor: `container/pompeu.def` (Apptainer) instala `requirements.lock`.
