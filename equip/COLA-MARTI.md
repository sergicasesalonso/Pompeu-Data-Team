# Cua del Martí

## Torn

**portàtil** — `portàtil` = treballo jo en local; `servidor` = treballa l'agent a `marti-agent`.

## Cua (ordenada)

«Agent» = la pot fer un agent sense supervisió contínua.

| # | Tasca | Crític 17-11 | Necessita dades | Agent |
|---|---|---|---|---|
| 1 | Confirmar l'hora de tancament del 17-11 i que tenim compte de Google per al formulari | Sí | no | no |
| 2 | Esborrany del document tècnic `PompeuDataTeam.md`: enfocament, actius usats i descartats, regles de possessió (pilota solta, duels, reinicis), validació, limitacions i reproducció. Font: `ARQUITECTURA.md` + leaderboard | Sí | no | sí |
| 3 | `configs/final.yaml` (= `rules_v0` mentre no es decideixi una altra cosa) i secció de reproducció al README: Python 3.11, `requirements.lock`, on van les dades, una sola ordre, sha256 esperat | Sí | no | sí |
| 4 | Verificar la reproducció en neta: venv nou amb Python 3.11 (o el contenidor `container/pompeu.def`) des d'un clon fresc, i comparar el sha256 | Sí | sí | sí |
| 5 | Revisar que cap fitxer derivat frame a frame entra al repo abans de cada push (`git log --all --name-only`) | Sí | no | sí |
| 6 | Generar `PompeuDataTeam.csv` final, passar `evaluation/validate_submission.py` sobre aquest fitxer exacte, i enviar el formulari (el 15 com a tard) | Sí | sí | no (l'enviament és meu) |
| 7 | Mirar el test intermitent `test_hsmm.py::test_throw_in_scene_labels_and_posteriors` (va fallar un cop amb 0,934 < 0,95) | No | no | sí |
| 8 | Passar el document tècnic a PDF i revisar-lo amb Sergi i Ander | Sí | no | no |
| 9 | Ajudar l'Ander amb A6 (conversor DFL o Metrica + `degrade.py`) si sobra temps | No | no (externes) | sí |

## Pendent d'avaluar

_(res encara)_ — configs o canvis que cal passar per `scripts/run_validation.py` en tornar.

## Preguntes

- Hora exacta de tancament del 17-11?
- Ens quedem amb `rules_v0` com a entrega final o Sergi vol provar l'híbrid / A9 abans de congelar?
- Substituïm el `README.md` de l'organització pel nostre, o hi afegim només una secció?
- Ander pot agafar A6 aquesta setmana? Amb quin dataset (DFL o Metrica sample)?
- Document tècnic en castellà, català o anglès?
