# Ablação de amostrador v0.9

- Campanha: `v0.9-bayes-sampler-ablation`
- Repetições: `3`
- Tamanho da amostra: `64`
- Commit: `23988d46b8effaef24fef7e285b2ea0e2d182eb5-dirty`

- Recomendação: **importance-v1**
| Sampler | MAE | TV | Excesso TV vs IID | Cobertura | Tempo (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `constrained-backtracking-v1` | 0.0248 | 0.1547 | +0.0059 | 0.9283 | 0.12 |
| `constrained-backtracking-short-v1` | 0.0248 | 0.1547 | +0.0059 | 0.9283 | 0.14 |
| `importance-v1` | 0.0241 | 0.1470 | -0.0018 | 0.9410 | 0.45 |
| `mcmc-v1` | falhou | constrained sampler emitted a fleet outside exact support |
