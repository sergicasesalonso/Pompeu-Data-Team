# Detector de balón muerto (A3)

`python scripts/deadball_report.py` — 7.2 s. Parámetros: `pompeu/events/deadball.py` PARAMS.

## Partido completo (sin GT)

| | |
|---|---|
| frames puntuados en `D` | 45.7 % (esperado 35–45 %) |
| tramos `D` | 97 |
| duración p10 / mediana / p90 | 7.9 / 22.4 / 106.3 s |
| P(D) media en frames puntuados | 0.457 |
| reinicios con equipo | 75 % |
| confianza media del equipo | 0.46 |

| tipo de reinicio | detectados | con salida del balón vista | Smart Tagging |
|---|---|---|---|
| throw_in | 30 | 21 | 26 |
| goal_kick | 7 | 4 | 10 |
| corner | 16 | 6 | 12 |
| free_kick | 38 | 0 | 5 |
| kickoff | 5 | 1 | 6 |

Fuentes del equipo: `kicker` 36, `none` 24, `tag` 14, `tag+geometry+kicker` 7, `geometry+kicker` 7, `geometry` 5, `tag+kicker` 2, `tag+geometry` 1

## Comprobación en la ventana pública (no sirve para elegir)

| | |
|---|---|
| F1 de `D` / F1 de vivo | 0.815 / 0.961 |
| tramos `D` en la GT / detectados | 4 / 3 |
| error del reinicio (frames, detectado − GT; a más de 5 s = perdido) | perdido, 27, -12, -33 |
