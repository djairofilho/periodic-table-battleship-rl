# Métodos: ataque

## Baselines

- `random_masked-v1`: tiro aleatório legal.
- `hunt-target-v1`: política estrutural clássica de busca por segmentos.

## Planejamento Bayesiano

O planejador Bayesiano trabalha com estado público:

1. Constrói uma população de frotas compatíveis com o histórico (hits/misses/sinks).
2. Converte a população em probabilidade por célula.
3. Escolhe tiro de maior cobertura (probabilidade, informação ou horizonte curto).

Na v0.7, foi usado `belief_probability_mc-v1` com validação separada em
`artifacts/v0.6-bayes-planner-validation` e `artifacts/v0.7-bayes-cross-topology-validation`.

## Destilação pública (CNN / GNN)

As estudantes recebem somente:

- observação pública de ataque (4 canais),
- máscara de ação legal,
- distribuição pública do professor.

Não recebem: posição de navios, recompensa privada, estado interno.

Em treino, monitoramos:

- acordo com professor,
- `valid_shots`,
- `auc_discovery`,
- comparação com `hunt-target-v1`.

## Críticas metodológicas

O ganho em um cenário não é critério único de promoção.
São necessários:

- consistência por `seed`,
- ganho pareado consistente entre topologias,
- critérios de gate explícitos no protocolo.

