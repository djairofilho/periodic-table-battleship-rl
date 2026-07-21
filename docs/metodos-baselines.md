# Baselines e contratos comparáveis

## O que é comparável

- Baseline `random_masked-v1`;
- Baseline `hunt-target-v1`;
- Políticas Bayesiano-probabilidade (`belief_probability_mc-v1`);
- Políticas distiladas (`bayesian-cnn-student-v1`, `bayesian-gnn-student-v1`).

## Métricas

O projeto separa métrica por papel:

- **Ataque:** menos `valid_shots` é melhor.
- **Posicionamento:** mais `valid_shots` (para o defensor sobreviver mais) é melhor.
- `auc_discovery` como suporte auxiliar para qualidade de sequência de descoberta.

## Contrato de comparação

- Seeds fixas para treino/validação.
- Tabelas JSON + CSV geradas por corrida.
- Bootstrap pareado com intervalo de 95%.
- Sem mistura entre validação e teste cego.

## Onde localizar o detalhe

Nos relatórios de versão (`docs/31-relatorio-v0.7.md` e `docs/28-validacao-bayesiana-multi-topologia-v0.7.md`)
há tabelas com intervalos de confiança e decisões de gate.

