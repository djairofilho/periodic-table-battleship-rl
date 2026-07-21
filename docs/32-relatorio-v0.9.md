# Relatorio v0.9: fechamento cientifico, decisao e rastreabilidade

## Decisao final de promocao

Nao houve promocao de nova politica nesta versao.

Evidencias:

- O planejamento Bayesiano (`belief_probability_mc-v1`) manteve ganho no smoke multi-topologia com `smoke` de 2 seeds.
- `bayesian-cnn-student-v1` / `bayesian-gnn-student-v1` foram treinadas em multi-seed, e nenhuma ficou melhor que `hunt-target-v1` em média.
- A decisão de ciclo é manter `hunt-target-v1` como baseline de produção e abrir v0.10 com nova tentativa de gate.

---

## 1) Calibracao do amostrador Bayesiano (`v0.9-bayes-sampler-calibration`)

Campanha: `artifacts/v0.9-bayes-sampler-calibration`

- seed: `7201`
- case_set: `extended`
- sample_count: `64`
- repetitions: `4`
- sampler: `constrained-backtracking-v1`

Metricas agregadas:

- MAE de ocupacao: **0.02666**
- RMSE de ocupacao: **0.03939**
- Total variation: **0.1612**
- Excesso de TV vs IID: **-0.01020**
- Cobertura de suporte: **0.9344**
- Massa fora do suporte: **0.0**

![Calibracao do amostrador v0.9](assets/belief-sampler-calibration-v0.9.png)

## 2) Ablacao de amostradores (`v0.9-bayes-sampler-ablation`)

Campanha: `artifacts/v0.9-bayes-sampler-ablation`

Variantes executadas:

- `constrained-backtracking-v1`
- `constrained-backtracking-short-v1`
- `importance-v1`
- `mcmc-v1` (falha por fora do suporte em alguns estados)

Recomendacao recomendada:

- `importance-v1` teve melhor excesso de TV (`-0.00177`), mas custo de runtime maior.

![Ablacao de samplers v0.9](assets/belief-sampler-ablation-v0.9.png)

## 3) Validacao multi-topologia (`v0.9-bayes-cross-topology-validation/smoke`)

Campanha: `artifacts/v0.9-bayes-cross-topology-validation/smoke`

Comparacao pareada Bayes x Hunt (metricas de tiros validos):

| Cenario | Bayes | Hunt | Diferenca (Bayes - Hunt) | IC95% |
| --- | ---: | ---: | ---: | --- |
| `battleship` | 35.0 | 51.0 | -16.0 | [-18.0, -14.0] |
| `dense-118` | 53.5 | 61.0 | -7.5 | [-8.0, -7.0] |
| `periodic-table-battleship` | 44.5 | 64.0 | -19.5 | [-33.0, -6.0] |

![Validacao Bayes x Hunt v0.9](assets/paired-valid-shots-v0.9.png)

Observação: validação executada em `smoke` (2 seeds, 1 episódio por seed), portanto insuficiente para gate de produção.

## 4) Dataset de demonstracoes (`v0.9-demonstrations`)

Campanha: `artifacts/v0.9-demonstrations`

- Total de decisoes: **375**
- Decisoes de treino: `battleship` 45, `dense-118` 33, `periodic-table-battleship` 76
- Decisoes de validacao: `battleship` 53, `dense-118` 78, `periodic-table-battleship` 90

Manifesto publico: `artifacts/v0.9-demonstrations/dataset-manifest-v0.9.json`

## 5) Distilacao Bayesiana multi-seed (`v0.9-bayesian-students`)

Campanha: `artifacts/v0.9-bayesian-students`

- seeds treino: `15001..15004`
- seeds validacao: `16001..16004`
- sample_count: `8`
- epochs: `4`
- arquiteturas: `cnn`, `gnn`
- hidden_dim: `16`
- soft_target_weight: `0.0`, `0.5`

Melhores validacoes por cenario:

- `battleship`: `gnn-h16-s0.00` com **55.25** tiros validos (base=63.75)
- `dense-118`: `cnn-h16-s0.50` com **79.00** tiros validos (base=65.50)
- `periodic-table-battleship`: `gnn-h16-s0.00` com **65.50** tiros validos (base=52.00)

![Estudantes v0.9 (multi-seed)](assets/bayesian-student-v0.9-valid-shots.png)

## 6) Estado de entrega

- Versao do pacote atualizada para `0.9.0`.
- Evidencias publicas preservadas em `artifacts/`.
- Rastreio reproduzivel em `docs/resultados.md`, `docs/reproducibilidade.md` e este relatorio.
- **Decisao de ciclo v0.9:** manter `hunt-target-v1`, sem self-play nesta fase.

## 7) Arquivos de referencia

- `artifacts/v0.9-bayes-sampler-calibration/belief-sampler-calibration-v0.9.json`
- `artifacts/v0.9-bayes-sampler-ablation/belief-sampler-ablation-v0.9.json`
- `artifacts/v0.9-bayes-cross-topology-validation/smoke/bayes-cross-topology-v0.9.json`
- `artifacts/v0.9-demonstrations/dataset-manifest-v0.9.json`
- `artifacts/v0.9-bayesian-students/bayesian-student-v0.9-report.json`
- `artifacts/v0.9-bayesian-students/bayesian-student-v0.9-results.csv`
