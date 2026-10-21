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
| `pompeu/features/kinematics.py` | Velocidad, aceleración y rumbo con Savitzky–Golay (ventana 7, orden 2), sin cruzar huecos |
| `pompeu/models/baseline.py` | El baseline oficial sobre `Tracking` (lo reproduce exacto) |
| `pompeu/pipeline.py` | Registro `METHODS`: `config → etiquetas`. Si la config tiene `clean:`, limpia primero. Cada método nuevo se registra aquí |
| `pompeu/validate/harness.py` | Los 4 bancos y el leaderboard |
| `pompeu/validate/stress.py` | Borra el balón en la ventana pública (30/50/70 %, 20 semillas) con la distribución real de huecos |
| `pompeu/validate/bootstrap.py` | Intervalos 5–95 % sobre el código oficial de la métrica |
| `events/ decode/ external/ vision/` | Vacíos, para las etapas A2–A10 y la Línea B |

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
python -m pytest                       # todo (~40 s)
python -m pytest tests/test_io.py      # 0.2: carga del ATD (199.319 frames, 46 slots, balón 56,9 %)
python -m pytest tests/test_contract.py  # 0.3: invariantes de Tracking
python -m pytest tests/test_baseline.py  # reproduce los 4 baselines oficiales byte a byte
python -m pytest tests/test_validate.py  # 0.4: stress, bootstrap y arnés
python -m pytest tests/test_clean.py     # A1: cada paso de limpieza en casos sintéticos y en el partido
```

Los tests que necesitan `data/` se saltan solos si no están los datos.

## Reglas del repositorio

- Nunca se sube `data/` ni nada derivado (`cache/`, `*.npz`, `*.parquet`, `results/*` salvo `leaderboard.csv` y `cleaning_report.md`). El repositorio será público.
- Dependencias: se edita `requirements.in` y se regenera `requirements.lock` con `pip freeze` en un venv limpio.
- Contenedor: `container/pompeu.def` (Apptainer) instala `requirements.lock`.
