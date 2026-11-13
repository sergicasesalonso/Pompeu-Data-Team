# Lectura crítica de `paper/PompeuDataTeam.md` — només propostes (12-11, nit)

No he reescrit el paper. Cada punt diu on és i què proposo. Comprovat contra codi, `results/leaderboard.csv`, `equip/ESTAT.md`, `EVALUATION.md` i `SUBMISSION.md`. Sense dades del partit.

## A. Afirmacions que contradiuen el que sabem (corregir abans d'entregar)

1. **§6 diu que `val` és «DFL J03WN1, Metrica 2»**, però `ESTAT.md` diu que Metrica 2 no s'ha pogut descarregar: tots els números de `val` (coverage sweep, sensibilitat de `min_possession_s`) són només de DFL J03WN1. Dir-ho: «`val` = una partida DFL (Metrica 2 no disponible)» o, si es descarrega, repetir. Comprovar també que Metrica 1 i 3 sí que hi són a la torre.
2. **«spellF1 differences below about 0.04 are noise (EVALUATION.md)» (§7)**: `EVALUATION.md` (l. 43 i 72) diu 0,01 per a una partida amb moltes frames i 0,04 *al partit complet* dels organitzadors. No és el soroll del nostre banc extern. Proposo treure la frase o substituir-la pels nostres intervals de bootstrap (columnes `external_spell_lo/hi`), que és el criteri de la regla de decisió.
3. **§1 «Every choice is made on a bench of other matches»** xoca amb §6: `min_possession_s` = 5 s i `dead_weight` surten de la finestra pública, i `a3_segment_llr` = 3 es va triar amb la finestra estressada. Suavitzar a «all model comparisons are made on…» i deixar la resta a §6.
4. **§6 «The 5 s value … a description of the label convention rather than a fit of any method»**: la nostra sensibilitat mostra que el llindar canvia el rànquing dels tres primers (0 s: `rules_v0` 0,315 vs `hsmm_v1` 0,266; 8 s: 0,351 vs 0,354). La frase és massa defensiva. Millor: «matched to the public reference's possession rate; since it moves the ranking we report the 2–8 s range and select a configuration that holds across it». Això depèn de la decisió de Martí sobre `min_possession_s`: no la prenc.
5. **§5 «The HSMM weights are being fitted … `configs/hsmm_v1_fit.yaml` does not exist yet»** i la fila `hsmm_v1_fit` de §7: text de treball. O hi ha fit o s'esborra; el document final no pot parlar de coses «in progress».
6. **§6, fila «Ball pseudo-labels (B1) → training the ball detector (B2)»**: si B2 no entra a l'entrega, aquest dataset no s'ha fet servir per a res del CSV. SUBMISSION.md demana «every labelled dataset you used». Dir explícitament «not used by the submitted file» (o treure'l a un paràgraf de «explored but not in the final») segons la decisió sobre la Línia B. A §1 i §2 igual: «Line B … in the final configuration: yes/no».

## B. Afirmacions sense suport visible al repo (marcar «to verify» o citar el script)

Verificades contra el codi: paràmetres de A1 (40 m/s, 3 frames, 0,5 s, 0,2 m en 10 s, 0,5 m en 1 s), slots 24 i 46 (docstring de `clean/players.py`), paràmetres de la degradació (41 %, 105 s, 0,12 %, 10–40 m, 5 cm), recomptes de possessions al partit complet (594, 250, 238 a `full_poss`).

Sense font al que he llegit:
- «Of 2,043 ball jumps above 40 m/s, 1,911 are removed» (§3): cal el script i la sortida (`cleaning_report.py`?). I «majority voting was already absorbing them» és una interpretació, no una mesura.
- «the ATD velocity column is empty on 89% of frames» (§4).
- «Smart Tagging … zone codes do not vote», «mean live period 40 s, mean dead 20 s», els pesos de cada senyal de l'A3 i els paràmetres de l'HSMM (§5): comprovar amb `deadball.PARAMS` i `hsmm.PARAMS` i, si coincideixen, deixar-ho; si no, corregir. No ho he fet línia per línia.
- «17 possessions in 4.0 live minutes» (§6) surt de la GT pública: és una dada de la finestra pública usada per a la convenció d'etiquetes. Ja és coherent amb el que diu el paper, però cal que Martí confirmi que això és acceptable per RULES.md (és una descripció, no entrenament).
- La conclusió de §2 «per-frame voting fragments possessions» està recolzada amb el partit complet (594 vs 250), que no té GT. Més sòlid: banc extern, `baseline_smooth50_clean` 1367 possessions vs 692 reals (§7 ja ho té). Usar aquest.

## C. Longitud i claredat

- ~3.700 paraules; SUBMISSION.md diu «a few pages is plenty». Proposo tallar a ~2.000–2.500:
  - §5: la llista de cues de l'A3 amb pesos i les regles de reinici → resumir en 5–6 línies i remetre a `ARQUITECTURA.md` del repo públic;
  - §4 (jugadors): una frase per mecanisme;
  - §6: la llista de la degradació en un paràgraf; la taula de datasets sí que cal.
- §7: l'últim paràgraf («the leaderboard also holds an earlier, DFL-only bench…») és història interna; fora. I la història de l'«stressed public window» (§6) en una frase.
- §3 «Tried and dropped»: una línia diu «list the remaining dropped variants»; hi ha candidats ja mesurats (`deadball_vote50`, `baseline_smooth25/125`, `hsmm_v1_rules`, ATD+ amb broadcast) amb xifres al leaderboard o a `ESTAT.md`.
- Consistència: escriure sempre «ATD+» (ara hi ha «improved tracking» i «ATD+» barrejats), «spellF1» igual a tot arreu i decimals amb punt. Les dues mesures de cobertura (59,3 → 63,1 al relleu antic; 59,2 → 62,9 a `ESTAT.md`) no s'han de barrejar; el paper ja fa servir `ESTAT.md`.
- La capçalera «DRAFT» i tots els «**TODO**» han de desaparèixer; fer un `grep TODO` abans del PDF.

## D. Què falta per a l'entrega del 17-11 (ordenat)

1. Decidir la configuració final (la regla de decisió de `CLAUDE.md`): amb els números actuals `rules_v0` és la candidata segura. Falten les files de `rules_v0`, `hsmm_v1`, `hsmm_v1_rules` al banc `test` nou (`bash scripts/tower_validate.sh` a la torre).
2. Omplir la taula de §7 i la frase del resultat de §1 només amb files del leaderboard (sense `+dirty`).
3. Decidir `min_possession_s` i `dead_weight` i redactar-ho (§6).
4. Decidir si la Línia B entra. Si no: reescriure §1/§2/§4 com a «explored, not in the submitted file» amb els números de cobertura (+3 punts ≈ +0,01) com a justificació. Això és un resultat net, no un fracàs.
5. sha256 final a §9 i al README (avui `bfd5ff16…` per a `rules_v0`), regenerat en un clon net després de la congelació (dl 16, 12:00).
6. A10: política final d'etiquetes (§5) i les limitacions finals (§8: anàlisi d'errors; ara és TODO).
7. Exportar a PDF o deixar el Markdown (SUBMISSION.md accepta tots dos) i comprovar que cobreix els sis apartats: enfocament (§1–2), actius (§3), dades d'entrenament (§6), regles (§5), limitacions (§8), reproducció (§9). Està tot; només falta que cap tingui TODO.
8. Revisar que el text no parla de com treballem (agents, Claude, torns): ara no hi és.
9. Comprovar al repo públic: cap dada, `docs/` i `CLAUDE.md` fora, requirements.lock, README coherent amb §9.

## E. Coses que estan bé (no tocar)

Taula de datasets amb llicències, dir què s'ha descartat, la nota sobre per què Metrica 3 és el de test, l'avís que les xifres absolutes de l'extern només es comparen amb el mateix `min_possession_s`, i la secció de neteja (cada regla amb el seu paràmetre).

## F. Actualització (12-11, nit)

- Corregits al paper (commit 0e100d1): descripció de `val` (només J03WN1), la frase del 0,04 i la del llindar de 5 s; «every choice» → «every comparison between methods». Pendent de Martí (no tocat): tallar el paper, Línia B sí/no, valor final de `min_possession_s` i `dead_weight`.
- **Verificats contra `PARAMS` del codi** (de B, «sense font»): A3 (`speed_ref` 1,4; `out_llr` 8; `absent_llr` 0,5; `still_spot_llr` 2 / `still_llr` 0,5; set piece 3 s +1; `goal_llr` 2; `pre_kickoff_llr` 8; kick −4 durant 1,5 s, 4 m/s, 0,5 s, 2 m; vida mitjana 40 s i 20 s; `min_live_gap_s` 5), A4 (3 m/s, 30°, 2 m, 80/20 amb `pitch_control_weight` 0,2; control 1,5 m / 0,6 s), `rules_v0` (3 s, 15 m/s, `min_team_confidence` 0,3 ≡ P(home) fora de 0,35–0,65) i HSMM (medianes 8 s / 20 s, mínims 0,2 / 1 s, 0,3 / 0,7, 0,98, 4 m, 2,5 m, `tag_tau_s` 5, `a3_segment_llr` 3). Tot coincideix amb §5. No he comprovat «ball more than 0.5 m outside the pitch» (a `deadball.py`, no és a `PARAMS`).
- Segueixen sense font: «1.911 de 2.043 salts» i «89 % de velocitat buida».

## G. Revisió independent dels canvis d'aquesta nit (12-11) i correccions

Una segona lectura, independent de la meva, va trobar errors meus. Corregits:
- Paper: la limitació nova citava la finestra pública estressada (0,506 → 0,294), cosa que contradiu §6 («no longer used»); ara cita el banc extern (0,438 amb 20 % de pilota absent → 0,387 amb 50 %). «Clearly first» → les diferències (0,017 i 0,013, sense intervals). S'explica per què la fila del 41 % de la taula de cobertura (0,396 / 0,390) no és la columna de 5 s de la de `min_possession_s` (0,400 / 0,386): el sweep és niat i amb una log-normal de forats. «Graded» → «in-half» frames a la taula de cobertura. §1 ara remet a §5 (`a3_segment_llr`) i §6. Coma que faltava a §2. «to check that two runs match» → «to verify».
- **Nou, per a §7:** la fila de `hsmm_v1_rules` del leaderboard (`766b069`) té `external_matches = J03WMX,J03WN1,J03WPY`: inclou J03WN1 (de `val`) i no Metrica 3. **No és del split `test`** i no pot anar a la taula tal com és: cal tornar-la a executar (`tower_validate.sh` ja la inclou).
- Un test meu (`test_atdplus_configs_do_not_clean_again`) no comprovava res: mirava claus de primer nivell i `clean:` va dins de `pipeline:`. Arreglat amb `yaml.safe_load`, i un test nou demostra que la guarda pot fallar.
