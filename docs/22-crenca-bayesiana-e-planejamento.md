# Crença Bayesiana e planejamento aproximado v0.6

## Propósito

Esta etapa cria referências explicáveis para o ataque. Em vez de uma rede
neural receber somente o tabuleiro achatado, uma crença mantém frotas legais
compatíveis com o histórico público e estima a ocupação de cada célula.

O estado de crença usa exclusivamente:

- acertos ativos;
- células de navios já afundados;
- água;
- geometria e máscara de ações válidas.

Nenhuma API lê `AttackEnv._fleet`, IDs de navio privados ou ocupação oculta.

## Dois níveis de cálculo

| Nível | Método | Garantia | Uso |
| --- | --- | --- | --- |
| Microtabuleiro | Enumeração completa | Crença exata, com limite explícito de frotas | Oráculo e teste de aproximações |
| Tabuleiro real | `constrained-backtracking-v1` | Toda amostra é compatível; frequência não é posterior exato | Planejadores rápidos e validação |

A enumeração exata interrompe com `CompatibleFleetLimitError` ao atingir
`max_fleets`; ela nunca passa a amostrar silenciosamente. O oráculo de
microtabuleiro está descrito em
[21-oraculo-exato-microtabuleiro.md](21-oraculo-exato-microtabuleiro.md).

O Monte Carlo escolhe primeiro o acerto ainda não explicado com menos
explicações possíveis, atribui uma colocação compatível a um navio restante e
depois completa a frota por backtracking. O relatório registra amostras,
restarts, backtracks e limite de nós. Como a ordem de propostas induz viés,
`posterior_exact` é sempre `false` nesta versão.

## Políticas avaliadas

Para uma população finita de frotas compatíveis, seja `p(c)` a fração de
amostras que ocupa a célula `c`.

| Política | Escolha | Objetivo declarado |
| --- | --- | --- |
| `belief_probability_mc-v1` | maior `p(c)` | Aumentar a chance imediata de acerto |
| `belief_information_mc-v1` | maior entropia binária de `p(c)` | Separar hipóteses de frota |
| `belief_horizon2_mc-v1` | maior número esperado de acertos em dois tiros | Planejar um horizonte curto |

Para a política informacional, o escore é:

```text
I(c) = -p(c) log p(c) - (1 - p(c)) log(1 - p(c))
```

Ela não afirma minimizar tiros até a vitória. O horizonte 2 também é um
surrogate: ele maximiza segmentos descobertos esperados, não resolve o POMDP
completo.

## Piloto de validação

O piloto usa apenas as seeds de validação `8601` a `8605`, 16 amostras por
decisão e uma seed de demonstração separada (`8701`). O teste cego não foi
aberto. Menos tiros é melhor.

| Política | Tiros válidos médios | Desvio entre seeds | Diferença pareada vs. hunt-target |
| --- | ---: | ---: | ---: |
| `belief_probability_mc-v1` | 43,20 | 9,58 | -29,80 |
| `belief_information_mc-v1` | 91,20 | 8,64 | +18,20 |
| `belief_horizon2_mc-v1` | 82,60 | 14,50 | +9,60 |
| `hunt_target-v1` | 73,00 | 12,00 | referência |

![Comparação de validação](../artifacts/v0.6-bayes-planner-validation/belief-policy-comparison.png)

O maior-probabilidade venceu `hunt-target` em todas as cinco seeds deste
piloto. Isso é uma hipótese promissora, não uma promoção: o orçamento é
pequeno, o amostrador é aproximado e a decisão precisa de uma campanha de
validação maior e de registro de candidata antes de qualquer teste cego.

As outras duas políticas pioraram. Maximizar apenas informação, em particular,
adiou tiros com alta chance de acerto.

![Padrões de tiros na demonstração](../artifacts/v0.6-bayes-planner-validation/belief-demo-heatmaps.png)

O segundo gráfico é somente uma demonstração com seed separada. Ele ilustra
que probabilidade, informação e horizonte curto criam padrões de busca
distintos; não é evidência estatística.

## Reprodução

Com as dependências visuais instaladas:

```powershell
uv run --extra visual python scripts/run_belief_planner_pilot.py --sample-count 16 --seed-count 5
uv run pytest tests/belief tests/experiments/test_belief_evaluation.py
```

Os resultados ficam em `runs/v0.6-bayes-planner-validation/` e os gráficos e
o relatório público em `artifacts/v0.6-bayes-planner-validation/`.
