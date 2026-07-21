# Resultados e decisões

Esta página é o resumo de leitura rápida para `README` e release.

## Ranking de destaque (v0.7)

| Política | Melhor métrica reportada | Contexto | Status |
| --- | ---: | --- | --- |
| `belief_probability_mc-v1` (Bayesiano) | 41,40 tiros | ataque | forte, mas sem requisito de generalização completo |
| `hunt-target-v1` | 73,00 tiros | ataque | baseline de referência estável |
| `bayesian-cnn-student-v1` | 62,00 a 70,50 | ataque | não promovida |
| `bayesian-gnn-student-v1` | 52,00 a 64,00 | ataque | não promovida |

## Decisão de promoção (v0.7)

- Bayesiano melhora em relação ao `hunt-target` em cenários específicos.
- Os intervalos de confiança não foram favoráveis em `battleship` e tabela periódica
  com consistência de gate.
- As estudantes CNN/GNN não alcançaram ganho robusto em múltiplos cenários com as sementes
  exigidas pelo protocolo.

## Evidências negativas (importantes)

- Em alguns cenários pontuais houve melhora de `valid_shots`, porém sem
  robustez pareada e sem repetição de seed suficiente para promoção.
- A amostragem Bayesiana ainda conserva viés pequeno, controlável por
  validação micro e por análise de cobertura.
- Self-play ficou reservado para a próxima etapa por decisão explícita no plano.

## Gráficos principais

![Comparação Bayesiano x Hunt por topologia](assets/paired-valid-shots-v0.7.png)

![Comparação estudantes públicos](assets/student-valid-shots-v0.7.png)

![Desvio do amostrador no microtabuleiro](assets/belief-sampler-calibration-v0.7.png)

## O que não avançou

Sem `2/3` dos cenários consistentes e sem margem de gate, a campanha v0.7 foi concluída como
**validação, não promoção**.

