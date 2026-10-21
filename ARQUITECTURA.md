# Arquitectura del código (Pompeu Data Team)

Todo nuestro código está en `pompeu/`. `evaluation/` y `starter/` son de la organización: se importan, nunca se editan. La métrica es siempre la de `evaluation/possession_metrics.py`.

## Flujo

```
data/ (ATD, XML)  →  pompeu.io.atd  →  Tracking  →  pipeline (config YAML)  →  etiquetas H/A/D
                                          ↑                                         ↓
                          Línea B escribe ATD+ (.npz)            validate.harness → results/leaderboard.csv
```

- **`Tracking`** (`pompeu/io/contract.py`) es el contrato entre las dos líneas. La Línea A solo lo lee y la Línea B solo lo escribe. Al crearse comprueba sus invariantes (formas, NaN coherentes, `ball_source`). Para cambiar de ATD a ATD+ basta con cambiar `tracking:` en el YAML.
- Coordenadas en metros (105,3 × 68,0), y la fila *i* es el frame de vídeo `frame0 + i`.

## Módulos

| Ruta | Qué hace |
|---|---|
| `pompeu/io/atd.py` | Parsea el `.txt` del ATD a numpy (`P[frame, slot, (x,y,s)]`, `B[frame, (x,y,s)]`). Tarda ~5 s la primera vez; luego se lee de la caché `.npz` |
| `pompeu/io/contract.py` | `Tracking` y `load_tracking("atd" \| ruta.npz)` |
| `pompeu/io/gt.py`, `tagging.py`, `homography.py` | GT pública, Smart Tagging (equipo → 0/1/−1) y homografía en metros |
| `pompeu/models/baseline.py` | El baseline oficial sobre `Tracking` (lo reproduce exacto) |
| `pompeu/pipeline.py` | Registro `METHODS`: `config → etiquetas`. Cada método nuevo se registra aquí |
| `pompeu/validate/harness.py` | Los 4 bancos y el leaderboard |
| `pompeu/validate/stress.py` | Borra el balón en la ventana pública (30/50/70 %, 20 semillas) con la distribución real de huecos |
| `pompeu/validate/bootstrap.py` | Intervalos 5–95 % sobre el código oficial de la métrica |
| `clean/ features/ events/ models/ decode/ external/ vision/` | Vacíos, para las etapas A1–A10 y la Línea B |

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
python -m pytest                       # todo (~15 s)
python -m pytest tests/test_io.py      # 0.2: carga del ATD (199.319 frames, 46 slots, balón 56,9 %)
python -m pytest tests/test_contract.py  # 0.3: invariantes de Tracking
python -m pytest tests/test_baseline.py  # reproduce los 4 baselines oficiales byte a byte
python -m pytest tests/test_validate.py  # 0.4: stress, bootstrap y arnés
```

Los tests que necesitan `data/` se saltan solos si no están los datos.

## Reglas del repositorio

- Nunca se sube `data/` ni nada derivado (`cache/`, `*.npz`, `*.parquet`, `results/*` salvo `leaderboard.csv`). El repositorio será público.
- Dependencias: se edita `requirements.in` y se regenera `requirements.lock` con `pip freeze` en un venv limpio.
- Contenedor: `container/pompeu.def` (Apptainer) instala `requirements.lock`.
