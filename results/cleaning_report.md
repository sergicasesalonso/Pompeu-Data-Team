# Informe de limpieza (A1)

Generado con `python scripts/cleaning_report.py` en 9.5 s. Configuración de limpieza: `{}` (vacío = valores por defecto de `pompeu/clean/__init__.py`).

## Balón

| | |
|---|---|
| frames con balón antes | 113,488 (56.9 %) |
| saltos > 40 m/s entre frames consecutivos, antes | 2,043 |
| spikes eliminados | 1,911 |
| puntos en fragmentos de < 3 frames eliminados | 4 |
| frames interpolados (huecos < 0,5 s, spline cúbico) | 26 |
| frames con balón después | 111,599 (56.0 %) |
| saltos > 40 m/s después | 10 |
| velocidad del balón (Savitzky–Golay) p50 / p95 / p99.9 | 5.4 / 20.6 / 40.9 m/s |

## Jugadores

| | |
|---|---|
| observaciones antes / después | 3,782,421 / 3,781,922 |
| frames congelados (< 0,2 m en 10 s con el resto moviéndose) | 41,830, acción `flag` |
| slots con más frames congelados | 24 (24,340), 46 (16,359), 4 (592) |
| tramos duplicados (< 0,5 m durante > 1 s) | 55, 2,236 observaciones eliminadas |
| tracklets / uniones (stitching) / cadenas | 2,266 / 302 / 1,964 |
| slots usados antes / después | 45 / 39 |
| mediana tracklet / cadena | 13.1 s / 15.1 s |
| frames rellenados dentro de una cadena (huecos <= 1 s) | 1,737 |
| jugadores visibles por equipo en frames puntuados, antes | 10.20 / 10.29 |
| jugadores visibles por equipo en frames puntuados, después | 10.20 / 10.28 |
| velocidad de jugadores p50 / p99 | 1.3 / 6.7 m/s |

## Notas

- Los slots 24 y 46 son los **porteros** (x ≈ 11 m / 94 m según la parte), no objetos fijos. Se congelan cuando salen de plano; por defecto esos frames se conservan con confianza 0,3 en vez de borrarse.
- La columna de velocidad del ATD no se usa: viene vacía en el 89 % de los frames.
- Efecto sobre el baseline (ventana ±50) en frames puntuados: 130 etiquetas cambian de 158,857.
