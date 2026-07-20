# Validação Bayesiana multi-topologia v0.7

Esta é a campanha de seleção da candidata Bayesiana. Ela compara
`belief_probability_mc-v1` com `hunt_target-v1` nos mesmos tabuleiros legais,
sob uma agenda fixa de validação. Não é teste cego e não deve ser usada como
resultado final da release.

## Protocolo congelado

| Item | Valor |
| --- | --- |
| Schema | `bayes-cross-topology-validation-v1` |
| Topologias | `battleship`, `dense-118`, `periodic-table-battleship` |
| Seeds | `8801` a `8810` |
| Episódios por seed | 1 |
| Amostras Monte Carlo por decisão | 16 |
| Métricas | tiros válidos e AUC de descoberta |
| Comparação | diferença pareada por seed, IC bootstrap percentil bilateral de 95% |
| Teste cego usado | não |

As duas políticas recebem a mesma seed de ambiente. Assim, cada par enfrenta a
mesma frota legal. Os IDs de execução são diferentes por desenho; o pareamento
é auditado pela agenda comum de seeds e pelo mesmo número de episódios em cada
seed.

O método Monte Carlo continua sendo `constrained-backtracking-v1`: cada frota
amostrada é compatível com o histórico público, mas a frequência observada não
é declarada posterior exata. A campanha mede desempenho, não corrige esse viés;
a calibração contra o micro-oráculo é uma etapa separada.

## Promoção

A política será considerada candidata apenas se o intervalo de 95% de
`Bayes − hunt-target` em tiros válidos estiver integralmente abaixo de zero nas
topologias declaradas. A regra é deliberadamente conservadora. Ela não abre o
teste cego: aprovação apenas autoriza registrar uma candidata congelada para a
próxima etapa.

## Execução

```powershell
uv run --extra visual python scripts/run_bayesian_cross_topology_validation.py
```

Para um check rápido, ainda exclusivamente de validação:

```powershell
uv run --extra visual python scripts/run_bayesian_cross_topology_validation.py --smoke
```

Os manifests por política ficam em
`runs/v0.7-bayes-cross-topology-validation/`; o JSON, a tabela Markdown e o
gráfico ficam em `artifacts/v0.7-bayes-cross-topology-validation/`.
