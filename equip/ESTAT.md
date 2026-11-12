# Estat

- **Entrega:** 17-11-2026, hora per confirmar. Congelació: dl 16-11, 12:00.
- **Entrega actual:** `configs/final.yaml` (= `rules_v0`): `python scripts/make_submission.py --config configs/final.yaml --out PompeuDataTeam.csv` → sha256 `bfd5ff16006b1e5c50371abeb528195cdc7a60397ec8bed90a8fccc468269f41`. Verificat en un clon net (Python 3.11, cache buida).
- **Entorn:** Python 3.11 + `requirements.lock`.
- **Tests:** 147 passen, 1 saltat (amb dades); sense dades també passen tots. El 12-11 a la nit s'hi han afegit tests sintètics nous (`tests/test_line_b_guards.py`, `tests/test_check_paper.py`); la suite completa encara no s'ha tornat a passar.

## Línia B (Martí)

| Etapa | Estat |
|---|---|
| B0 vídeo | ✅ 199.319 frames, homografia vàlida, 99,91 % de la pilota dins de la imatge, àudio AAC 48 kHz |
| B1 pseudo-etiquetes | ✅ 88.937 (70.449 train / 18.488 val), a `cache/` |
| B8 broadcast | ✅ alineació fiable (jugadors 0,29 m). Pilota: soroll entre trackers ~0,7 m + cua de pilotes ràpides/aèries i falses (no és un desfasament: lag 0 és el millor) |
| B3 fusió | ✅ `augment_ball()`. ATD+ v1 (cobertura 59,2 → 62,9 %, hash 9faa52d2…) i v2 amb filtres (59,2 → 62,1 %, hash dc03433e…). Configs experimentals `*_atdplus.yaml` (llegeixen v1) |
| B9 determinisme | proposta a `equip/B9-PROPOSTA.md`; ordre estable a `broadcast.align` i hash amb NaN/-0.0 canònics fets (12-11, tests sintètics). Per verificar a la torre: ATD+ ×2 amb el mateix hash, i que els hashes de v1/v2 no canvien |

## Document

`paper/PompeuDataTeam.md`: ple, amb TODO als resultats finals.

## Valor de la cobertura (scripts/coverage_sweep.py, external val dfl:J03WN1, 5 llavors)

| pilota absent | rules_v0 spellF1 | hsmm_v1 spellF1 |
|---|---|---|
| 50 % | 0,387 | 0,377 |
| 41 % | 0,396 | 0,390 |
| 37 % | 0,409 | 0,400 |
| 30 % | 0,423 | 0,410 |
| 20 % | 0,438 | 0,421 |

+1 punt de cobertura ≈ +0,0025 spellF1 (rules_v0 +0,0032 a 41→37 %). El broadcast (+3 punts) val ~+0,01: dins del soroll. Falta Metrica 2 (no s'ha pogut descarregar).

## Paràmetres que depenien de la ventana pública

- `dead_weight` (degrade.py): mesurat sense GT amb l'A3 (`scripts/ball_missing_by_state.py`): pilota absent 74,7 % mort vs 12,0 % viu, ràtio 6,2.
- `min_possession_s` (`scripts/min_possession_sensitivity.py`, val): mou el rànquing dels 3 primers; rules_v0 és el més robust. Decidir la config final mirant el rang 2–8 s.
- `a3_segment_llr`: pendent de fer servir el valor ajustat a val.
