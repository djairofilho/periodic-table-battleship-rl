# Calibração do amostrador Bayesiano no microtabuleiro

## Objetivo

O planejador usa `constrained-backtracking-v1` para construir frotas
compatíveis com o histórico público. Compatibilidade não implica que a
frequência das frotas seja o posterior exato. Esta calibração mede a diferença
no único caso em que ela pode ser calculada por enumeração completa: o
microtabuleiro 3×3 com um navio de comprimento 2.

Esse espaço contém as mesmas 12 frotas físicas uniformes do oráculo de
programação dinâmica. Os históricos públicos são fixos e não usam seeds
de validação nem o inventário de teste cego:

| Histórico | Evidência pública coberta |
| --- | --- |
| `prior` | nenhuma chamada |
| `active-hit-center` | acerto ainda não afundado |
| `miss-center` | água |
| `hit-corner-miss-center` | acerto e água |
| `sunk-top-edge` | navio anunciado como afundado |
| `two-ship-*` | estresse com dois navios, ainda enumerável exatamente |

## Métricas

Para cada histórico, a referência é `exact_belief`: distribuição uniforme em
todas as frotas compatíveis. Para cada repetição independente, o runner mede:

- MAE, RMSE e erro máximo entre os mapas de ocupação;
- distância de variação total (TV) entre a distribuição empírica de frotas e
  a crença exata;
- cobertura do suporte exato, isto é, quantas frotas compatíveis apareceram;
- massa de amostras fora do suporte exato, que deve ser exatamente zero.

O relatório também calcula a TV de um controle IID extraído da própria crença
exata com o mesmo orçamento. A diferença `TV proposta - TV IID` ajuda a
separar a variância finita inevitável da discrepância adicional da proposta de
backtracking. O controle não transforma a proposta em posterior exato.

A TV não separa viés de proposta e ruído finito de Monte Carlo. Portanto,
mesmo com discrepância baixa, o campo `posterior_exact` permanece `false` no
amostrador de produção. A calibração também não afirma que o prior uniforme
seja idêntico ao gerador `random_legal-v1` dos cenários grandes.

## Resultado fixado

Com 1.024 amostras e 32 repetições por histórico, nenhuma amostra saiu do
suporte exato e todos os suportes foram integralmente visitados. No caso de um
navio, a TV observada acompanha o controle IID. No estresse de dois navios sem
chamadas, a TV média da proposta foi `0,1287`, contra `0,1173` do controle IID:
excesso de `0,0115`. É uma evidência mensurada de discrepância adicional,
pequena neste microcaso, e não uma justificativa para marcar o posterior de
produção como exato.

## Reprodução

```powershell
uv run --extra visual python scripts/run_belief_sampler_calibration.py
uv run pytest tests/belief/test_calibration.py
```

O runner grava JSON com todas as repetições, uma tabela CSV e o gráfico em
`artifacts/v0.7-bayes-sampler-calibration/`. O relatório sempre declara
`blind_test_used: false`.

![Discrepância e cobertura no microtabuleiro](../artifacts/v0.7-bayes-sampler-calibration/belief-sampler-calibration.png)
