# Periodic Table Battleship RL

## Resumo em 60 segundos

- **Objetivo:** treinar agentes de reinforcement learning para a Batalha Naval
  em dois mundos: `battleship` e `periodic-table-battleship`.
- **Contribuição atual:** baseline `hunt-target-v1` segue sendo difícil de superar
  e o projeto mantém trilha de validação robusta (separação entre treino,
  validação e teste cego).
- **Status científico:** o planejador Bayesiano é robusto em várias validações em
  micro e em alguns cenários, mas ainda sem promoção para teste cego no ciclo.

## Comece aqui

1. [Visão geral e decisões atuais](visao-geral.md)
2. [Especificação do jogo e cenários](jogo.md)
3. [Métodos (Bayesiano e NN)](metodos-ataque.md)
4. [Resultados e interpretações positivas/negativas](resultados.md)
5. [Reprodutibilidade](reproducibilidade.md)
6. [Roadmap técnico](roadmap-0.8-0.9.md)

## Resultado mais recente (v0.7)

| Métrica | Melhor valor | Referência | Decisão |
| --- | ---: | ---: | --- |
| Planejador Bayesiano (ataque) | 41,40 tiros | 73,00 (`hunt-target-v1`) | Excelente em média, porém só 1 de 3 cenários com IC favorável |
| CNN/GNN distilada (ataque) | 52,00 a 70,50 tiros | 48,50 a 65,50 (`hunt-target-v1`) | Não houve promoção multi-seed |
| Self-play de posicionamento | Não aplicado na promoção | - | Mantido como próximo candidato experimental |

## Artefatos gráficos principais

![Validação Bayesiana por topologia](assets/paired-valid-shots-v0.7.png)

![Calibração do amostrador no microtabuleiro](assets/belief-sampler-calibration-v0.7.png)

