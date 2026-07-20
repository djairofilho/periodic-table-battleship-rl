# Avaliação cross-topology v0.4

## Objetivo e proteção contra comparação acidental

O avaliador padrão de PPO continua exigindo que a topologia gravada no
checkpoint seja idêntica à topologia avaliada. A transferência entre tabuleiros
usa exclusivamente a API explícita `run_cross_topology_ppo_attack_evaluation`.
Ela valida o checkpoint contra a topologia de origem e persiste, para cada
célula, origem, destino, números de células e ações, hashes de checkpoint e
metadados, e o protocolo `cross-topology-public-observation-v1`.

O agente recebe somente observação pública e máscara de ações. A API rejeita
topologias com contagens de ações diferentes, pois índices de saída de uma
política não são transferíveis com segurança nesse caso.

## Protocolo executado

Foram usados os checkpoints de ataque v0.3 escolhidos na validação para a seed
fixa de treino 3101. As 100 seeds de teste são as mesmas da agenda cega v0.3,
de 5101 a 5200. A diagonal é um controle same-topology; as demais células são
transferência explícita. Assim, os valores são descritivos para um checkpoint
representativo por topologia, e não substituem uma análise multi-seed de
políticas treinadas.

| Treino | Teste | Média de tiros válidos |
| --- | --- | ---: |
| clássico | clássico | 94,80 |
| clássico | `dense-118` | 111,52 |
| clássico | tabela periódica | 109,88 |
| `dense-118` | clássico | 94,77 |
| `dense-118` | `dense-118` | 111,70 |
| `dense-118` | tabela periódica | 111,29 |
| tabela periódica | clássico | 94,21 |
| tabela periódica | `dense-118` | 111,87 |
| tabela periódica | tabela periódica | 111,02 |

Todos os episódios terminaram em vitória. Os resultados sugerem que, nesse
orçamento, o comportamento aprendido transfere pouco de forma específica: as
diferenças entre origem e destino são pequenas e não superam a baseline
hunt-target já registrada em v0.3. Isso não é um teste causal de
generalização, porque ainda falta repetir a matriz para as cinco seeds de
treino de cada topologia e calcular intervalos por seed de política.

## Reprodução

```powershell
uv run --extra train python scripts/run_cross_topology_v0_3.py
```

O comando grava nove runs em `runs/v0.4-cross-topology/` e o relatório público
em [`cross-topology-report.json`](../artifacts/v0.4-cross-topology/cross-topology-report.json),
[`cross-topology-matrix.csv`](../artifacts/v0.4-cross-topology/cross-topology-matrix.csv)
e [`cross-topology-summary.md`](../artifacts/v0.4-cross-topology/cross-topology-summary.md).
