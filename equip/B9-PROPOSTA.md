# B9 — proposta de determinisme de la Línia B

Estat: **proposta**, sense executar (el servidor no té dades). Tot el que diu «mesura» va a la torre.

## Què ja és determinista (llegit al codi)

- `fuse.augment_ball`: només numpy/scipy en float64, `lexsort` (estable) amb desempats explícits, `connected_components`; sense llavors ni paral·lelisme. Mateixes entrades → mateixos arrays.
- `broadcast.align` / `ball_candidates`: numpy pur, sense llavors. Veure els dos riscos de sota.
- `scripts/build_atd_plus.py` ja imprimeix un sha256 dels arrays (`tracking_hash`); v1 i v2 tenen hash anotat a `ESTAT.md`.
- B1 (`pseudolabels.py`): píxels arrodonits a 0,1 px, divisió train/val per blocs de frame (sense aleatorietat).

## Riscos concrets a corregir (són a fitxers meus; ho faria jo)

1. **Empats amb ordenació no estable** a `broadcast.align`: `np.argsort(-counts)` (modes de l'offset) fa servir quicksort per defecte. Amb empats de recompte, l'ordre pot canviar entre versions/CPU de numpy (kernels SIMD) i canvia l'ordre de `offsets`, i amb ell quin offset guanya un empat a `min(fits, ...)`. Fix: `kind="stable"`, i desempat explícit pel valor de l'offset.
2. **Hash sensible als bits de NaN**: `tracking_hash` fa `tobytes()` d'arrays amb NaN; un NaN pot tenir signe/payload diferent segons l'operació que el crea (`0/0` vs `np.nan`). Fix: canonitzar (`np.where(isnan(a), np.nan, a)`) abans de hashejar. Evita falsos «no coincideix».
3. **Llindars sobre valors sense arrodonir**: `fit <= max_fit_m`, `speed > max_speed_ms`, `median > ref_max_m`, `_reach`... Un canvi d'1e-12 pot canviar una decisió. A la CPU és estable, però un detector de GPU canvia al darrer bit. Fix: arrodonir les **entrades del tracker** (veure «Contracte»), no cada llindar.

## Contracte d'arrodoniment (abans del tracker)

- `detect_ball` → `x_px, y_px` a 0,1 px (com B1), `conf` a 3 decimals, ordre `frame, -conf, x_px, y_px` (desempat total).
- `video_candidates_to_metres`: arrodonir `x_m, y_m` a 0,01 m just després de la projecció, i `conf` a 3 decimals; el broadcast igual a `ball_candidates`. Així `fuse` rep sempre una graella discreta.
- Limitació honesta: arrodonir no elimina tots els canvis (un valor a prop d'un tall d'arrodoniment encara pot saltar). Per això la prova de sota mesura quants frames canvien.

## Detector (Ander, B2) — condicions que li demanaria

`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.backends.cudnn.deterministic=True`, `benchmark=False`, llavors fixes, `num_workers=0` o `worker_init_fn` fixat, mida de batch fixa (el batch pot canviar la reducció), sense AMP si canvia l'ordre de reducció, versions de torch/CUDA/cuDNN anotades al README. Entrenament: pesos publicats amb hash, o reentrenament determinista (regla 10).

## Proves (per ordre de cost; totes a la torre, només mètriques)

1. **ATD+ dues vegades, mateixa màquina** (CPU): `build_atd_plus.py` ×2 → mateix `tracking_hash`. (Ja fet per a v1/v2; repetir amb el detector.)
2. **Sensibilitat**: afegir soroll uniforme de ±0,05 px i ±0,0005 de conf a les deteccions (llavor fixa), passar-les per l'arrodoniment + `augment_ball`, i informar **% de frames puntuats amb pilota que canvien** i quants frames del CSV final canvien. Si és ~0, l'arrodoniment basta; si no, llindars amb marge.
3. **Detector, dues execucions a la mateixa GPU**: mateix hash de les deteccions arrodonides.
4. **GPU vs CPU** (inferència): fracció de deteccions que canvien després d'arrodonir i mateix `tracking_hash` de l'ATD+. Si el detector no és bit-idèntic entre GPU i CPU, decidir: (a) publicar l'ATD+ amb hash (no es pot: és derivat del partit), o (b) exigir la GPU/versions anotades. Això és decisió del Martí/equip.
5. **Clon net + `make_submission.py`**: mateix sha256 del CSV (criteri 5 de la regla de decisió).

## Decisió que cal prendre

Només si el detector omple prou forats (veure «Valor de la cobertura» a `ESTAT.md`: +1 punt ≈ +0,0025 d'spellF1) val la pena el cost de B9. Si no, la Línia B no entra a l'entrega i B9 es redueix a «`rules_v0` regenera el mateix sha256», que ja està verificat. Proposo decidir-ho dg 15 amb el número de cobertura del detector a la mà, abans de dedicar-hi la GPU.
