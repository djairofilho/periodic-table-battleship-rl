# Self-play contra planejador Bayesiano v0.6

Esta etapa integra o planejador `belief_probability_mc-v1` à liga como um
atacante congelado com identidade própria. O ledger não o declara como PPO:
ele registra a estratégia, o número de amostras Monte Carlo e o SHA-256 de um
contrato declarativo local.

## Escopo do piloto

O piloto executa uma única atualização de posicionamento no cenário clássico.
O posicionador MaskablePPO aprende contra uma mistura de peso 1,0 formada pelo
atacante Bayesiano congelado. Em seguida, ele é avaliado contra dois atacantes
fixos: o planejador Bayesiano e `hunt-target`.

| Item | Decisão |
| --- | --- |
| Split | Somente validação |
| Seeds | `8611`, `8612`, `8613` |
| Planejador | Maior ocupação Monte Carlo, 8 amostras por decisão por padrão |
| Liga | Um bootstrap Bayesiano, um bootstrap de posicionamento e uma atualização de posicionador |
| Promoção | Nunca automática; permanece `not-decided` |
| Teste cego | Não usado |

O planejador recebe apenas observação pública e máscara de ações. Para avaliar
uma frota, `PublicAttackPolicyEvaluator` cria um `AttackEnv` com a frota
privada, mas a política só vê os canais públicos retornados pelo ambiente.

## Reprodução

Requer as dependências de treino e visualização:

```powershell
uv run --extra train --extra visual python scripts/run_bayesian_selfplay_pilot.py --timesteps 64 --sample-count 8
uv run --extra train --extra visual pytest tests/selfplay
```

O comando grava o ledger em
`runs/v0.6-bayesian-selfplay-validation/` e o relatório, incluindo gráfico,
em `artifacts/v0.6-bayesian-selfplay-validation/`. Checkpoints grandes ficam em
`.local-runs/`; o ledger persiste somente IDs, hashes, seeds e ancestrais.

## Interpretação

Este piloto valida o acoplamento e a proveniência, não demonstra que um
posicionador é melhor. Qualquer campanha de promoção deve pré-registrar
múltiplas seeds, comparar contra os mesmos avaliadores congelados e só então
considerar um teste cego separado.
