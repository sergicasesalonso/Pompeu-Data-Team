# Rasgos por frame (A2)

`python scripts/build_features.py` — 203 columnas, 199,319 frames. Matriz en `cache\features_v1_0b569aa50a.parquet` (no se sube a git). % NaN medido sobre los frames puntuados.

## ball (20 columnas)

| columna | % NaN |
|---|---|
| `ball_present` | 0.0 |
| `ball_interp` | 0.0 |
| `ball_speed` | 40.8 |
| `ball_acc` | 40.8 |
| `ball_turn_deg` | 48.3 |
| `ball_still` | 40.8 |
| `ball_still_s` | 40.8 |
| `ball_out` | 40.8 |
| `ball_to_goal_line_min` | 40.8 |
| `ball_to_touchline_min` | 40.8 |
| `ball_edge_dist` | 40.8 |
| `ball_dist_centre` | 40.8 |
| `ball_dist_corner` | 40.8 |
| `ball_dist_penalty` | 40.8 |
| `ball_x_home` | 40.8 |
| `ball_y_home` | 40.8 |
| `ball_x_home_last` | 0.0 |
| `ball_gap_age_s` | 0.0 |
| `ball_gap_left_s` | 0.0 |
| `ball_gap_len_s` | 0.0 |

## proximity (20 columnas)

| columna | % NaN |
|---|---|
| `d1_H` | 40.9 |
| `d2_H` | 40.9 |
| `d3_H` | 40.9 |
| `n2_H` | 40.8 |
| `n5_H` | 40.8 |
| `n10_H` | 40.8 |
| `near_speed_H` | 40.9 |
| `d1_A` | 40.8 |
| `d2_A` | 40.8 |
| `d3_A` | 40.8 |
| `n2_A` | 40.8 |
| `n5_A` | 40.8 |
| `n10_A` | 40.8 |
| `near_speed_A` | 40.8 |
| `d1_diff` | 40.9 |
| `nearest_team` | 40.9 |
| `near_streak_s_H` | 40.9 |
| `near_streak_fwd_s_H` | 40.9 |
| `near_streak_s_A` | 40.9 |
| `near_streak_fwd_s_A` | 40.9 |

## pitch_control (4 columnas)

| columna | % NaN |
|---|---|
| `pc_ball_H` | 40.8 |
| `pc_ring_mean_H` | 40.8 |
| `pc_ring_min_H` | 40.8 |
| `pc_ring_max_H` | 40.8 |

## collective (31 columnas)

| columna | % NaN |
|---|---|
| `n_vis_H` | 0.0 |
| `cx_H` | 0.5 |
| `cy_H` | 0.5 |
| `cvx_H` | 0.5 |
| `cvy_H` | 0.5 |
| `width_H` | 0.9 |
| `depth_H` | 0.9 |
| `def_line_H` | 0.9 |
| `front_H` | 0.5 |
| `compact_H` | 0.5 |
| `n_recover_H` | 0.0 |
| `n_press_H` | 0.0 |
| `speed_mean_H` | 0.5 |
| `n_vis_A` | 0.0 |
| `cx_A` | 0.0 |
| `cy_A` | 0.0 |
| `cvx_A` | 0.0 |
| `cvy_A` | 0.0 |
| `width_A` | 0.6 |
| `depth_A` | 0.6 |
| `def_line_A` | 0.6 |
| `front_A` | 0.0 |
| `compact_A` | 0.0 |
| `n_recover_A` | 0.0 |
| `n_press_A` | 0.0 |
| `speed_mean_A` | 0.0 |
| `field_speed_mean` | 0.0 |
| `field_speed_median` | 0.0 |
| `stopped_frac` | 0.0 |
| `centroid_gap` | 0.5 |
| `line_gap` | 0.9 |

## context (20 columnas)

| columna | % NaN |
|---|---|
| `half` | 0.0 |
| `minute` | 0.0 |
| `st_active` | 0.0 |
| `st_vote` | 0.0 |
| `st_vote_n` | 0.0 |
| `st_zone_team` | 0.0 |
| `sp_team` | 0.0 |
| `goal_active` | 0.0 |
| `st_prev_start_s` | 0.0 |
| `st_prev_vote` | 0.0 |
| `st_prev_end_s` | 0.0 |
| `st_next_start_s` | 0.0 |
| `sp_prev_s` | 0.0 |
| `sp_next_s` | 1.1 |
| `sp_throw_in` | 0.0 |
| `sp_corner` | 0.0 |
| `sp_goal_kick` | 0.0 |
| `sp_free_kick` | 0.0 |
| `sp_kickoff_goal` | 0.0 |
| `sp_kickoff_start` | 0.0 |

## windows (108 columnas)

| columna | % NaN |
|---|---|
| `ball_speed_w0.5_mean` | 39.8 |
| `ball_acc_w0.5_mean` | 39.8 |
| `d1_H_w0.5_mean` | 39.9 |
| `d1_A_w0.5_mean` | 39.8 |
| `d1_diff_w0.5_mean` | 39.9 |
| `n5_H_w0.5_mean` | 39.8 |
| `n5_A_w0.5_mean` | 39.8 |
| `pc_ball_H_w0.5_mean` | 39.8 |
| `field_speed_mean_w0.5_mean` | 0.0 |
| `stopped_frac_w0.5_mean` | 0.0 |
| `cvx_H_w0.5_mean` | 0.5 |
| `cvx_A_w0.5_mean` | 0.0 |
| `ball_speed_w0.5_min` | 39.8 |
| `ball_acc_w0.5_min` | 39.8 |
| `d1_H_w0.5_min` | 39.9 |
| `d1_A_w0.5_min` | 39.8 |
| `d1_diff_w0.5_min` | 39.9 |
| `n5_H_w0.5_min` | 39.8 |
| `n5_A_w0.5_min` | 39.8 |
| `pc_ball_H_w0.5_min` | 39.8 |
| `field_speed_mean_w0.5_min` | 0.0 |
| `stopped_frac_w0.5_min` | 0.0 |
| `cvx_H_w0.5_min` | 0.5 |
| `cvx_A_w0.5_min` | 0.0 |
| `ball_speed_w0.5_max` | 39.8 |
| `ball_acc_w0.5_max` | 39.8 |
| `d1_H_w0.5_max` | 39.9 |
| `d1_A_w0.5_max` | 39.8 |
| `d1_diff_w0.5_max` | 39.9 |
| `n5_H_w0.5_max` | 39.8 |
| `n5_A_w0.5_max` | 39.8 |
| `pc_ball_H_w0.5_max` | 39.8 |
| `field_speed_mean_w0.5_max` | 0.0 |
| `stopped_frac_w0.5_max` | 0.0 |
| `cvx_H_w0.5_max` | 0.5 |
| `cvx_A_w0.5_max` | 0.0 |
| `ball_speed_w2_mean` | 36.6 |
| `ball_acc_w2_mean` | 36.6 |
| `d1_H_w2_mean` | 36.7 |
| `d1_A_w2_mean` | 36.6 |
| `d1_diff_w2_mean` | 36.7 |
| `n5_H_w2_mean` | 36.6 |
| `n5_A_w2_mean` | 36.6 |
| `pc_ball_H_w2_mean` | 36.6 |
| `field_speed_mean_w2_mean` | 0.0 |
| `stopped_frac_w2_mean` | 0.0 |
| `cvx_H_w2_mean` | 0.5 |
| `cvx_A_w2_mean` | 0.0 |
| `ball_speed_w2_min` | 36.6 |
| `ball_acc_w2_min` | 36.6 |
| `d1_H_w2_min` | 36.7 |
| `d1_A_w2_min` | 36.6 |
| `d1_diff_w2_min` | 36.7 |
| `n5_H_w2_min` | 36.6 |
| `n5_A_w2_min` | 36.6 |
| `pc_ball_H_w2_min` | 36.6 |
| `field_speed_mean_w2_min` | 0.0 |
| `stopped_frac_w2_min` | 0.0 |
| `cvx_H_w2_min` | 0.5 |
| `cvx_A_w2_min` | 0.0 |
| `ball_speed_w2_max` | 36.6 |
| `ball_acc_w2_max` | 36.6 |
| `d1_H_w2_max` | 36.7 |
| `d1_A_w2_max` | 36.6 |
| `d1_diff_w2_max` | 36.7 |
| `n5_H_w2_max` | 36.6 |
| `n5_A_w2_max` | 36.6 |
| `pc_ball_H_w2_max` | 36.6 |
| `field_speed_mean_w2_max` | 0.0 |
| `stopped_frac_w2_max` | 0.0 |
| `cvx_H_w2_max` | 0.5 |
| `cvx_A_w2_max` | 0.0 |
| `ball_speed_w5_mean` | 30.4 |
| `ball_acc_w5_mean` | 30.4 |
| `d1_H_w5_mean` | 30.6 |
| `d1_A_w5_mean` | 30.4 |
| `d1_diff_w5_mean` | 30.6 |
| `n5_H_w5_mean` | 30.4 |
| `n5_A_w5_mean` | 30.4 |
| `pc_ball_H_w5_mean` | 30.4 |
| `field_speed_mean_w5_mean` | 0.0 |
| `stopped_frac_w5_mean` | 0.0 |
| `cvx_H_w5_mean` | 0.4 |
| `cvx_A_w5_mean` | 0.0 |
| `ball_speed_w5_min` | 30.4 |
| `ball_acc_w5_min` | 30.4 |
| `d1_H_w5_min` | 30.6 |
| `d1_A_w5_min` | 30.4 |
| `d1_diff_w5_min` | 30.6 |
| `n5_H_w5_min` | 30.4 |
| `n5_A_w5_min` | 30.4 |
| `pc_ball_H_w5_min` | 30.4 |
| `field_speed_mean_w5_min` | 0.0 |
| `stopped_frac_w5_min` | 0.0 |
| `cvx_H_w5_min` | 0.4 |
| `cvx_A_w5_min` | 0.0 |
| `ball_speed_w5_max` | 30.4 |
| `ball_acc_w5_max` | 30.4 |
| `d1_H_w5_max` | 30.6 |
| `d1_A_w5_max` | 30.4 |
| `d1_diff_w5_max` | 30.6 |
| `n5_H_w5_max` | 30.4 |
| `n5_A_w5_max` | 30.4 |
| `pc_ball_H_w5_max` | 30.4 |
| `field_speed_mean_w5_max` | 0.0 |
| `stopped_frac_w5_max` | 0.0 |
| `cvx_H_w5_max` | 0.4 |
| `cvx_A_w5_max` | 0.0 |
