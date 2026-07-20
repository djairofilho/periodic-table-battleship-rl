# Relatório v0.6: oráculo, crença pública e habilitação GPU

## Pergunta e protocolo

Esta release avalia planejamento sob incerteza para Batalha Naval, sem expor
navios ocultos ao agente. A escolha de política ocorre em validação; o teste
cego permanece fechado. O ciclo contém três níveis:

1. um oráculo de programação dinâmica em microtabuleiro, onde a solução é
   exata;
2. crença por enumeração ou Monte Carlo no tabuleiro completo, onde a
   aproximação é declarada;
3. ablação neural, self-play de integração e benchmark de hardware, sem usar
   velocidade como evidência de qualidade.

## Oráculo exato

No tabuleiro 3 × 3, com um único navio de tamanho 2, há 12 configurações
legais sob prior uniforme. A programação dinâmica sobre o estado público
encontra 4,50 tiros esperados. O arquivo
[`oracle-report.json`](../artifacts/v0.6-micro-oracle/oracle-report.json)
registra valores por ação e 2.353 estados de crença memorizados.

| Política | Tiros esperados | Regret contra o oráculo |
| --- | ---: | ---: |
| Oráculo DP | 4,50 | 0,00 |
| Posterior guloso | 4,50 | 0,00 |
| Hunt-target | 4,94 | 0,44 |
| Aleatória mascarada | 6,67 | 2,17 |

![Oráculo no microtabuleiro](../artifacts/v0.6-micro-oracle/oracle-comparison.png)

Q-learning e SARSA usam as mesmas 12 frotas e a mesma observação pública
ternária. Após 5.000 episódios em cada uma de quatro seeds, a avaliação exata
das políticas gulosas obteve 6,235 tiros para Q-learning e 6,442 para SARSA.
Os dois aprendem acima do aleatório, porém não chegam ao oráculo. A
[comparação tabular](25-q-learning-sarsa-micro-oraculo.md) contém a curva,
hiperparâmetros e regret por seed.

## Planejamento Bayesiano no jogo completo

No cenário clássico 10 × 10, o amostrador
`constrained-backtracking-v1` produz frotas compatíveis somente com o
histórico público. O planejador de probabilidade usa 64 amostras por decisão;
portanto é uma aproximação Monte Carlo, não um posterior exato.

| Política, cinco seeds de validação | Média de tiros válidos | Desvio-padrão |
| --- | ---: | ---: |
| Maior probabilidade | 41,40 | 2,41 |
| Hunt-target | 73,00 | 12,00 |
| Horizonte 2 | 81,60 | 24,95 |
| Informação | 93,40 | 9,91 |

Menos tiros é melhor. A alternativa de maior probabilidade superou
`hunt-target` nesta validação, enquanto as heurísticas de informação e
horizonte curto não. Os dados, mapas e resumo estão em
[`runs/v0.6-bayes-planner-validation`](../runs/v0.6-bayes-planner-validation)
e [`artifacts/v0.6-bayes-planner-validation`](../artifacts/v0.6-bayes-planner-validation).

![Políticas Bayesiana e baseline](../artifacts/v0.6-bayes-planner-validation/belief-policy-comparison.png)

## Rede híbrida e self-play

A CNN híbrida recebeu apenas dois planos derivados da crença pública:
probabilidade de ocupação e entropia. Em três seeds de treino e três de
validação, com 1.024 passos, ela obteve 97,22 tiros contra 96,89 do controle.
Não há ganho e a candidata não é promovida.

O piloto de self-play treinou um posicionador PPO por 64 passos contra o
planejador Bayesiano congelado. Ele elevou o número de tiros do atacante
Bayesiano de 42,67 para 46,00, mas reduziu o número contra `hunt-target` de
62,33 para 57,67. O resultado serve para validar a liga e seus registros; não
constitui uma promoção.

![Ablação da rede híbrida](../artifacts/v0.6-hybrid-belief-pilot/hybrid-belief-ablation.png)

## GPU e decisão de escala

A `.venv-cuda` independente validou PyTorch `2.13.0+cu130` numa GTX 1650 de
4 GiB. O benchmark mede apenas forward, backward e Adam com o mesmo lote e
seed em CPU e CUDA.

| Arquitetura | CPU atualizações/s | CUDA atualizações/s | CUDA/CPU |
| --- | ---: | ---: | ---: |
| CNN | 41,80 | 248,31 | 5,94× |
| DQN MLP | 361,39 | 314,81 | 0,87× |
| GNN | 51,40 | 163,46 | 3,18× |

O ambiente e a medição estão prontos, mas uma campanha GPU grande exigia uma
candidata neural promovida em validação. Como a rede híbrida foi rejeitada, a
campanha não é executada nesta release e o teste cego continua intacto. Essa
é uma decisão experimental, não uma limitação de hardware.

## Limitações e próximo gate

Os ganhos do planejador foram medidos somente em validação clássica, e o
posterior completo é aproximado. Antes de promover qualquer resultado geral,
será necessário pré-registrar orçamento e seeds, comparar nos três cenários e
abrir um conjunto de teste novo, uma única vez. A GPU fica disponível para uma
futura candidata neural que cumpra esse gate.
