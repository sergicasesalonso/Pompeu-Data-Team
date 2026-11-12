# Estat de l'equip (Pompeu Data Team)

Actualitzat: 2026-11-12 (anàlisi completa de la branca `marti` = `origin/sergi` + fitxers d'equip)

## Entrega

- **Primera entrega: 17-11-2026, hora per confirmar.** SUBMISSION.md només diu la data; l'URL del formulari arriba per correu. El pla diu enviar-ho **el 15 com a tard**.
- Pel formulari: `PompeuDataTeam.csv`, URL del repo públic i `PompeuDataTeam.pdf`/`.md` (document tècnic). Cal un compte de Google. Una resposta per equip (editable).
- Segons el pla, **avui (12-11) és el dia de congelació** de la configuració.

## Requisits de l'entrega

| Requisit | Estat | Detall |
|---|---|---|
| Fitxer de 199.319 files vàlid | ✅ | `rules_v0`: `validate_submission.py` OK, 0 `X`, 261 possessions de 13,8 s |
| Es regenera idèntic | ✅ (en local) | Dues execucions: sha256 `bfd5ff16006b1e5c50371abeb528195cdc7a60397ec8bed90a8fccc468269f41`. Falta provar-ho en una màquina neta / contenidor |
| Ordre única | ⚠️ | Existeix (`python scripts/make_submission.py --config configs/rules_v0.yaml --out PompeuDataTeam.csv`) però no hi ha `configs/final.yaml` ni surt al README |
| Dependències fixades | ⚠️ | `requirements.lock` fixat, però **només funciona amb Python 3.11** (amb 3.13 numpy 2.0.2 no té wheel i falla). No està escrit enlloc visible. `container/pompeu.def` existeix però no s'ha provat |
| Instruccions d'instal·lació i execució | ⚠️ | Són a `ARQUITECTURA.md` (castellà, intern). El `README.md` encara és el de l'organització (amb una línia retocada) |
| Document tècnic | ❌ | No existeix. És obligatori |
| Repo públic sense dades | ✅ | El repo ja és públic. Historial net: només `data/README.md`, `results/leaderboard.csv` i els tres `*_report.md` (recomptes) |

## Codi (`pompeu/`)

| Mòdul | Estat |
|---|---|
| `io/` (ATD, contracte `Tracking`, GT, tagging, homografia) | ✅ Funciona i té tests |
| `clean/` (A1) | ✅ Funciona; no millora el baseline per si sol |
| `features/` (A2, ~200 columnes) | ✅ Funciona. Falta `restarts.py` del pla (ho cobreix `events/deadball.py`) |
| `events/` (A3 balón muerto, A4 toques) | ✅ Funciona. Criteri de A3 (F1 > 0,90) no verificable sense `external` |
| `models/` (baseline, deadball_vote, rules_v0, hsmm_v1) | ✅ Funcionen. Falten `seqmodel` (A7), `ensemble`/`calibration` (A8) |
| `decode/` | ⚠️ `viterbi.py` fet; falten `mbr.py` (A9) i `grammar.py` |
| `validate/` | ⚠️ `public_clean`, `public_stress` i `full_match_sanity` funcionen; `external` és un placeholder |
| `external/` (A6) | ❌ Buit (només `__init__.py`) |
| `vision/` (Línia B) | ❌ Buit. Tampoc hi ha `scripts/slurm/` |

## Tests

`python -m pytest` amb les dades: **101 passen, 0 fallen** (1 min 54 s). Al primer cop sense dades, `test_hsmm.py::test_throw_in_scene_labels_and_posteriors` va fallar una vegada (0,934 < 0,95) i després ha passat 4 vegades seguides: possible test intermitent (probablement la primera compilació de numba). A vigilar.

## Leaderboard (`results/leaderboard.csv`)

`spellF1` decideix el rànquing; `macroF1` desempata. Banc més realista: `public_stress` (el partit real té ~41 % de pilota absent, entre 30 % i 50 %).

| Config | spellF1 clean | s30 | s50 | s70 | macroF1 clean | macroF1 s50 / s70 | possessions |
|---|---|---|---|---|---|---|---|
| **`rules_v0`** (entrega) | 0,506 | 0,466 | **0,360** | 0,294 | 0,843 | 0,734 / 0,684 | 250 |
| `deadball_vote125` | **0,529** | **0,470** | 0,341 | 0,289 | 0,844 | 0,727 / 0,701 | 284 |
| `hsmm_v1` | 0,449 | 0,410 | 0,352 | **0,322** | 0,835 | **0,767 / 0,743** | 238 |
| `baseline_smooth50` (terra publicat: 0,2433 al partit) | 0,326 | 0,261 | 0,218 | 0,155 | 0,621 | 0,527 / 0,434 | 594 |

**On perd punts:**
1. **Quan falta la pilota.** `rules_v0` passa de 0,506 (clean) a 0,294 (70 % sense pilota). Sense pilota no hi ha toques i la possessió no canvia: al 70 % detecta 8,8 possessions de 17.
2. **Segmentació curta.** 250 possessions al partit, fora del rang 300–400 de la guia (`full_ok = False` a totes les files). Sergi argumenta que aquest rang no és coherent amb un 45 % de `D`; no ho podem confirmar sense `external`.
3. **% de `D` = 45,7 %**, just per sobre del 35–45 % esperat.
4. Els intervals 5–95 % se solapen entre `rules_v0`, `deadball_vote125` i `hsmm_v1`: **cap diferència entre ells és significativa** amb 17 possessions.
5. Gairebé totes les files són `+dirty` (sense commit abans de llançar).

## Pla vs fet

| Etapa | Pla | Fet |
|---|---|---|
| 0 Infraestructura | ✅ | ✅ (sense `scripts/slurm/` ni MareNostrum) |
| A1–A5 | ✅ | ✅ |
| A6 dades externes + degradació | Banc que decideix | ❌ |
| A7 model seqüencial (GPU) | | ❌ |
| A8 calibració/ensemble | | ❌ |
| A9 MBR | | ❌ |
| A10 política final d'etiquetes | | ⚠️ Parcial: ja hi ha 0 `X` |
| Línia B (B0–B9) | | ❌ Res |
| Etapa final (README, doc, verificació neta) | 13–16 nov | ❌ |

## Riscos

1. **Document tècnic inexistent** a 5 dies. Bloquejant.
2. **Reproducció en una màquina neta no provada**; el requisit de Python 3.11 no està escrit al README.
3. **No hi ha banc `external`**, així que triar entre `rules_v0` i `hsmm_v1` és una aposta. La regla de l'equip diu quedar-se amb `rules_v0`.
4. **Hora de tancament del 17-11 desconeguda**.
5. Dades completes a `data/` i ignorades per git, però el vídeo tàctic es diu `BAR-ATH_tactical-001.mp4` i `games.py` espera `BAR-ATH_tactical.mp4`: cal reanomenar-lo. No afecta la Línia A.
6. Canviar de model a última hora sense `external` pot fer baixar la nota real encara que millori `public_*`.
7. `README.md` de l'organització retocat: cal decidir si el substituïm pel nostre.

## Repartiment proposat

| Qui | Feina | Necessita dades |
|---|---|---|
| **Martí** | Entrega: document tècnic, README de reproducció, `configs/final.yaml`, verificació en màquina neta/contenidor, formulari | Doc i README: no. Verificació: sí |
| **Sergi** | Model: congelar la config final, revisar el doc tècnic (és el que coneix el codi), opcional: híbrid `rules_v0` amb pilota + `hsmm_v1` als forats, o A9 | Sí |
| **Ander** | A6 mínim: conversor DFL o Metrica sample data + `degrade.py` + banc `external` al harness, per poder decidir amb dades i no a cegues | No les del partit (dades externes públiques). Per calibrar la degradació: sí |
| Ningú abans del 17-11 | Línia B, A7, A8 (necessiten GPU/MareNostrum i no hi ha res fet) | Sí + GPU |

Detall de les tasques del Martí: `equip/COLA-MARTI.md`.
