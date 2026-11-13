# Cua del Martí

## Torn

**portàtil** — `portàtil` = treballa el Martí; `servidor` = treballa l'agent del servidor a `marti` (commits locals).

## Pendent d'avaluar

1. Recollir els commits locals del servidor (`git log origin/marti..marti` al servidor; cap = l'últim d'`equip/RELLEU.md`) i portar-los a `marti`. Comprovar que cap té trailers.
2. Decisions obertes (recomanacions al canal intern, missatge «decisio per-a-marti»):
   - Línia B a l'entrega: sí / no (recomanació: no; +3 punts de cobertura ≈ +0,01 spellF1, dins del soroll).
   - `min_possession_s` (recomanació: 5 s i publicar el rang 2–8 s).
   - `dead_weight`: 6,2 mesurat o 5,0 actual.
   - Tallar el paper a ~2.000–2.500 paraules (`equip/REVISIO-PAPER.md`, §C).
   - Etiquetar vídeo extern: recomanació no (missatges «etiquetar-a-ma» i «video-extern-etiquetat»).
3. A la torre (només mètriques):
   - `bash scripts/tower_validate.sh`: files de `rules_v0`, `hsmm_v1`, `hsmm_v1_rules` al split `test`, cap `+dirty`. La fila actual de `hsmm_v1_rules` (766b069) no és de `test`.
   - `python scripts/build_atd_plus.py` ×2 per a v1 i v2: mateix hash, i comparar-lo amb 9faa52d2… / dc03433e….
   - Recompte de candidats del broadcast amb velocitat NaN (`equip/REVISIO-LINIA-B.md`, punt 3).
   - Font de «1.911 de 2.043 salts» i «89 % de velocitat buida» del paper, o treure-les.
4. Suite completa amb dades (`python -m pytest`): al servidor, 163 tests, 0 errors, 23 saltats per falta de dades.

## Cua del servidor (sense dades del partit)

Feta la del 12-11 (B9, paper, revisió de la Línia B). Si torna a ser torn de servidor: el que digui el Martí.

## No tocar

- Fitxers de Sergi (`pompeu/models/`, `decode/`, `events/`, `features/`, `external/`, `validate/`, `requirements*`) i de l'Ander (`pompeu/vision/ball_det/`, `audio.py`).
- `configs/final.yaml` (l'entrega continua sent `rules_v0`).
