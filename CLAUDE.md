# CLAUDE.md

Projecte del Pompeu Data Team (Martí, Sergi i Ander) per al More Than A Hack 2027. L'arquitectura del codi és a `ARQUITECTURA.md`; l'estat de l'equip, a `equip/`.

## Regles

- **«me’n vaig»**: actualitza els tres fitxers d'`equip/` (`ESTAT.md`, `COLA-MARTI.md`, `RELLEU.md`), posa el torn a `servidor` a `COLA-MARTI.md`, fes commit i push a `marti` i digues-m'ho.
- **«torno»**: fes `git pull` de `origin/marti-agent`, revisa'n els canvis i executa primer el que hi hagi a «pendent d'avaluar» de `equip/COLA-MARTI.md`.
- **Mai facis commit de `data/`** ni de res derivat frame a frame (tracking, rasgos, etiquetes per frame, `cache/`, `*.npz`, `*.parquet`, `results/*` excepte `leaderboard.csv` i els `*_report.md`): el repo és públic i les condicions ho prohibeixen.
- **No toquis mai `main` ni `sergi` directament**: el que estigui bé hi arriba per PR.
- Els documents d'equip van a `equip/`, no a `docs/` (`/docs` és al `.gitignore`).
