# Piloto Bayesiano v0.6: validação

Este é um piloto de validação, sem acesso ao teste cego. As frotas do
planejador Monte Carlo são compatíveis com o histórico público, mas a
distribuição proposta por backtracking não é declarada como posterior exato.

- Seeds de validação: `[8601, 8602, 8603, 8604, 8605]`
- Seed de demonstração: `8701`
- Amostras por decisão: `64`
- Revisão de origem: `5588e3a3bb6c677663f9fd231ede5892211bc8d1`

| Política | Tiros válidos médios | Desvio entre seeds |
| --- | ---: | ---: |
| `belief_probability_mc-v1` | 41.40 | 2.41 |
| `belief_information_mc-v1` | 93.40 | 9.91 |
| `belief_horizon2_mc-v1` | 81.60 | 24.95 |
| `hunt_target-v1` | 73.00 | 12.00 |

![Comparação de políticas](belief-policy-comparison.png)

![Distribuição de tiros de demonstração](belief-demo-heatmaps.png)

A figura de demonstração usa seed separada e é ilustrativa, não
evidência para promoção.
