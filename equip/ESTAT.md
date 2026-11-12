# Estat de l'equip (Pompeu Data Team)

Actualitzat: 2026-11-12

## Entrega

- **Primera entrega: 17-11-2026, hora per confirmar** (SUBMISSION.md només diu la data; l'URL del formulari arriba per correu).
- Cal pujar: `PompeuDataTeam.csv`, URL del repo públic i `PompeuDataTeam.pdf`/`.md` (document tècnic).
- Validar el fitxer exacte abans de pujar-lo: `python evaluation/validate_submission.py PompeuDataTeam.csv`.

## Qui fa què

| Qui | Branca | Què |
|---|---|---|
| Sergi | `sergi` | Pipeline A1–A5 (neteja, rasgos, balón muerto, `rules_v0`, `hsmm_v1`) |
| Martí | `marti` | Per definir (veure `equip/COLA-MARTI.md`) |
| Ander | — | Per definir |

`main` i `sergi` no es toquen directament: tot hi arriba per PR.

## Millor resultat del leaderboard (`results/leaderboard.csv`)

Mètrica de rànquing: `spellF1`. Interval 5–95 % entre parèntesis.

| Config | clean | stress 30 % | stress 50 % | stress 70 % | possessions |
|---|---|---|---|---|---|
| **`rules_v0`** (entrega actual) | 0,506 | 0,466 | **0,360** | 0,294 | 250 |
| `deadball_vote125` | **0,529** | **0,470** | 0,341 | 0,289 | 284 |
| `hsmm_v1` | 0,449 | 0,410 | 0,352 | **0,322** | 238 |

- L'entrega de seguretat és `rules_v0` (commit `5fc5a1c`): `python scripts/make_submission.py --config configs/rules_v0.yaml`.
- Amb 17 possessions públiques els intervals se solapen: cap canvi es pot donar per acceptat sense el banc `external` (A6).

## Què falta (resum)

- A6: banc `external` (partits externs degradats) per decidir de debò.
- A7: emissió apresa de qui té la pilota quan no es veu.
- A9: MBR sobre els posteriors de l'HSMM.
- Document tècnic de l'entrega.
