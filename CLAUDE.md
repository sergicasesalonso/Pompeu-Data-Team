# CLAUDE.md

Projecte del Pompeu Data Team per al More Than A Hack 2027. L'arquitectura del codi és a `ARQUITECTURA.md`. El portàtil i el servidor treballen tots dos a la branca `marti`.

## On va cada cosa

- `equip/` (es puja): només el mínim operatiu: `ESTAT.md`, `COLA-MARTI.md` (torn i «pendent d'avaluar") i `RELLEU.md`.
- `docs/` (local, al `.gitignore`): el pla, l'estat detallat, la cua i tot el que expliqui com treballem. No es puja mai.

## Regles

- **«me’n vaig»**: actualitza els tres fitxers d'`equip/` i els de `docs/` que calgui, posa el torn a `servidor`, fes commit i push a `marti` i digues-m'ho.
- **«torno»**: fes `git pull` de `origin/marti`, revisa'n els canvis i executa primer el que hi hagi a «pendent d'avaluar».
- **Mai facis commit de `data/`** ni de res derivat frame a frame (tracking, rasgos, etiquetes per frame, `cache/`, `*.npz`, `*.parquet`, `results/*` excepte `leaderboard.csv` i els `*_report.md`): el repo és públic i les condicions ho prohibeixen.
- No llegeixis ni imprimeixis el contingut de `data/` ni sortides per frame; treballa amb mètriques agregades.
- Mai facis servir dades d'esdeveniments o de possessió del BAR-ATH de cap font externa (RULES.md).
- **No toquis mai `main` ni `sergi` directament**: el que estigui bé hi arriba per PR.
- Mai afegeixis `Co-Authored-By` de Claude (ni cap atribució a Claude) als missatges de commit.
