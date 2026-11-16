# Clasificador vivo/muerto (A3, paso 3)

`python scripts/train_deadball_gbm.py` — LightGBM entrenado con `train` (1425888 filas, 185 rasgos), parada temprana con `val`, mejor iteración 50.

F1 de **muerto** por frame en `test` (partidos degradados, semillas [0, 1]):

| partido | semilla | LightGBM solo | reglas A3 | A3 + LightGBM |
|---|---|---|---|---|
| dfl:J03WMX | 0 | 0.952 | 0.910 | 0.942 |
| dfl:J03WMX | 1 | 0.954 | 0.908 | 0.940 |
| dfl:J03WPY | 0 | 0.967 | 0.915 | 0.960 |
| dfl:J03WPY | 1 | 0.959 | 0.915 | 0.957 |
| metrica:3 | 0 | 0.775 | 0.765 | 0.787 |
| metrica:3 | 1 | 0.776 | 0.761 | 0.793 |
| **total** | | **0.914** | **0.877** | **0.911** |

Rasgos más importantes (ganancia): `ball_speed_w0.5_mean`, `ball_speed_w0.5_min`, `field_speed_mean_w5_mean`, `ball_speed`, `ball_speed_w5_min`, `field_speed_mean_w2_mean`, `ball_edge_dist`, `ball_speed_w2_mean`, `stopped_frac_w5_min`, `width_H`, `ball_speed_w5_mean`, `ball_x_home_last`, `field_speed_median`, `cvx_A_w5_min`, `ball_gap_len_s`
