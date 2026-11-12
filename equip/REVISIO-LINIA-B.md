# Revisió de `fuse.py`, `broadcast.py` i `build_atd_plus.py` — només propostes

Llegits sense executar res (12-11, nit). Després, amb el sí de Martí, s'han aplicat els punts 1 i 2; la resta segueixen sent propostes. Res d'això afecta `rules_v0`. Ordenat per valor.

## Correcció / determinisme
1. `broadcast.align`: `np.argsort(-counts)` no és estable → ordre dels offsets pot variar amb empats. Usar `kind="stable"`. **Fet (12-11).**
2. `build_atd_plus.tracking_hash`: NaN amb bits diferents donen hashes diferents. Canonitzar NaN abans de `tobytes()`. **Fet (12-11).**
3. `broadcast.ball_candidates`: `_ball_speed` és una diferència central sobre **tota** la línia de temps del broadcast, així que és NaN als dos extrems de la gravació i a **qualsevol frame amb un veí sense pilota**; el filtre `fast = isfinite(v) & (v > max)` deixa passar tots aquests frames sense filtrar. Pot ser molt més que un parell de files: és cada vora de cada forat de pilota del broadcast. (Als talls de càmera, en canvi, la diferència creua el tall i probablement sí que es filtra.) Proposta: a la vora d'un forat, fer servir la diferència d'un sol costat (endavant o enrere) en lloc de NaN. A mesurar a la torre: quants candidats tenen `v` NaN (només el recompte).

## Claredat / manteniment
4. `fuse.augment_ball(..., H=None)`: el paràmetre `H` s'ignora (`del H`). Treure'l, o documentar que és per simetria i que els tests no el passin.
5. `fuse`: tots els candidats acceptats tenen `ball_source = SRC_VIDEO`, també els del broadcast; el contracte no té codi «broadcast». Mentre no es canviï (acord amb Sergi), l'informe ja separa per font (`accepted_per_source`). Cal dir-ho al document (ja hi és) i vigilar que cap model de Sergi pesi diferent «video».
6. `build_atd_plus.build`: la neteja es fa a `load_tracking("atd")` + `clean_tracking(raw, {})`; les configs `*_atdplus.yaml` no han de tenir `clean:`. Un test que ho comprovi (fitxers `configs/*_atdplus.yaml` sense la clau `clean`) evitaria netejar dues vegades.
7. `broadcast._local_disagreement`: bucle Python per candidat amb `np.median`; O(candidats × finestra). Funciona (s'ha executat), però si es reutilitza amb molts candidats del detector (8/frame) caldria vectoritzar o fer-ho per finestres lliscants.
8. `_votes`: `s + (e - s) // 2 % every` es llegeix com una expressió ambigua; posar parèntesis explícits (`s + ((e - s) // 2) % every`) o un comentari.

## Cobertura de tests (suggeriments, tests nous a `tests/`, sense dades)
- empat de recomptes a la votació d'offsets → ordre estable;
- `tracking_hash` igual per dos Tracking amb NaN de bits diferents;
- `augment_ball` idempotent: aplicar-lo dues vegades al resultat no afegeix res (cap candidat nou ni interpolació nova);
- `ball_candidates` amb `max_fit_m=0` (conf = 1) i amb `reference_xy` de forma incorrecta (ja llança `ValueError`).

## Verificat al codi, sense problema
- `cand_of_node[is_cand] = order[is_cand]` és correcte: els nodes candidats són els primers del concatenat, així que `order` dona l'índex original.
- `_remove_spikes` i `_short_new_fragments` mai toquen punts de l'ATD (`protected` / `has_old`).
- Cap llavor aleatòria ni paral·lelisme a `fuse.py` i `broadcast.py`.

## Fora dels meus fitxers (proposta per a Sergi, `pompeu/validate/harness.py`)
- `harness.run` crida `git_version()` a cada config i després afegeix la fila a `results/leaderboard.csv`, que és versionat: amb `run_validation.py --config a.yaml b.yaml`, la fila de `b` surt `+dirty` encara que l'arbre estigués net. Proposta: calcular `git_version()` una vegada a l'inici de `run_validation.py`, o excloure `results/leaderboard.csv` del test de brut. Mentrestant, `scripts/tower_validate.sh` fa un commit del leaderboard entre configs.
