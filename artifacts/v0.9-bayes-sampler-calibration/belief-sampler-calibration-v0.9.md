# Calibração do amostrador Bayesiano (v0.9)

- Campaign: `v0.9-bayes-sampler-calibration`
- Sampler: `constrained-backtracking-v1`
- Case set: `prior`

| Caso | Frotas reais | MAE ocupação | Excesso TV vs IID | Cobertura do suporte |
| --- | ---: | ---: | ---: | ---: |
| `prior` | 12 | 0.0356 | +0.0208 | 1.000 |
| `active-hit-center` | 4 | 0.0174 | -0.0430 | 1.000 |
| `miss-center` | 8 | 0.0234 | -0.0703 | 1.000 |
| `hit-corner-miss-center` | 2 | 0.0165 | -0.0586 | 1.000 |
| `sunk-top-edge` | 1 | 0.0000 | +0.0000 | 1.000 |
| `two-ship-prior` | 88 | 0.0503 | -0.0284 | 0.545 |
| `two-ship-active-hit-center` | 48 | 0.0451 | -0.0117 | 0.734 |
| `one-ship-edge-hit-top` | 3 | 0.0203 | +0.0273 | 1.000 |
| `one-ship-edge-miss-top` | 9 | 0.0500 | +0.0508 | 1.000 |
| `one-ship-corner-hit-opposite-miss` | 2 | 0.0139 | +0.0117 | 1.000 |
| `one-ship-center-hit-corner-miss` | 4 | 0.0139 | -0.0312 | 1.000 |
| `two-ship-corner-miss` | 32 | 0.0382 | -0.0078 | 0.867 |
| `two-ship-edge-miss` | 14 | 0.0221 | +0.0078 | 1.000 |

- Total de amostras por caso: `64` x `4` repetições
- Taxa média de cobertura: `0.9344`
- Módulo de completude: `0.9344`
