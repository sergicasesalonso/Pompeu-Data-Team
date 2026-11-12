# Estat

- **Entrega:** 17-11-2026, hora per confirmar. Congelació: dl 16-11, 12:00.
- **Entrega actual:** `configs/final.yaml` (= `rules_v0`): `python scripts/make_submission.py --config configs/final.yaml --out PompeuDataTeam.csv` → sha256 `bfd5ff16006b1e5c50371abeb528195cdc7a60397ec8bed90a8fccc468269f41`. Verificat en un clon net (Python 3.11, cache buida).
- **Entorn:** Python 3.11 + `requirements.lock`.
- **Tests:** 129 passen, 1 saltat.

## Línia B (Martí)

| Etapa | Estat |
|---|---|
| B0 vídeo | ✅ 199.319 frames, homografia vàlida, 99,91 % de la pilota dins de la imatge, àudio AAC 48 kHz |
| B1 pseudo-etiquetes | ✅ 88.937 (70.449 train / 18.488 val), a `cache/` |
| B8 broadcast | ✅ alineació fiable; +9,2 % dels frames sense pilota (cobertura 59,3 → 63,1 %) |
| B3 fusió | ✅ `augment_ball()` amb tests sintètics. ATD+ amb broadcast: en curs (`marti-atdplus`, portàtil) |
| B9 determinisme | pendent |

## Document

`paper/PompeuDataTeam.md`: ple, amb TODO als resultats finals.
