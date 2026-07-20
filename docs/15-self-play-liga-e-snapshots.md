# Self-play: liga, snapshots e avaliação estável

A issue [#40](https://github.com/djairofilho/periodic-table-battleship-rl/issues/40)
introduz uma base reproduzível para *self-play*. Aqui, *self-play* significa
treinar alternadamente o atacante contra posicionadores congelados e o
posicionador contra atacantes congelados. Não é uma única política jogando
contra si mesma: cada papel aprende uma decisão diferente e a avaliação precisa
continuar comparável enquanto os oponentes evoluem.

## Contrato de liga

O módulo `periodic_table_battleship_rl.selfplay` fornece:

- `SnapshotProvenance`: identidade portátil de checkpoint, com SHA-256, papel,
  cenário, run de origem e ancestrais, sem caminho local;
- `SnapshotLeague`: pool imutável por cenário e amostragem uniforme
  determinística do oponente;
- `SelfPlayCampaignConfig` e `AlternatingSelfPlaySchedule`: calendário
  alternado, budgets e seeds gerados deterministicamente;
- `FrozenEvaluationSuite`: baselines que não mudam junto com a liga;
- `SelfPlayCampaignRecord`: ledger append-only que só aceita snapshots com o
  oponente selecionado registrado como ancestral.

O ledger é JSON portátil. Ele permite retomar uma campanha sem trocar
silenciosamente o oponente de uma rodada histórica, e evita registrar caminhos
de checkpoint que só fazem sentido na máquina que treinou o modelo.

## Ambiente e runner acoplados

O módulo agora inclui o acoplamento mínimo, explicitamente separado dos
benchmarks v0.3/v0.4:

- `PlacementPolicyFleetSampler` transforma uma política de posicionamento
  congelada em gerador de frotas ocultas para o ambiente;
- `CoupledAttackEnv` é um `AttackEnv` que obtém a frota daquele gerador. A
  observação do atacante continua contendo somente topologia, acertos, navios
  afundados, erros e máscara de ações;
- `PublicAttackPolicyEvaluator` transforma uma política de ataque congelada
  em avaliador de posicionamento. A política recebe somente observação pública
  e máscara de ações a cada tiro;
- `CoupledSelfPlayRunner` resolve o snapshot sorteado pela liga, chama o
  treinador do papel alternado, calcula o SHA-256 do checkpoint e registra a
  rodada e seus resultados congelados.

O runner é deliberadamente agnóstico ao algoritmo de treino. Um adaptador de
experimento implementa `train_attacker` e `train_placer`; isso permite usar
PPO, DQN ou outro método sem mudar as regras de liga ou a proveniência.

O mapeamento de snapshots para objetos carregados é local e não é serializado.
O ledger contém somente IDs, hashes, seeds, budgets e ancestrais portáteis.

## Protocolo para o primeiro experimento acoplado

A campanha inicial deve usar:

| Campo | Decisão inicial |
| --- | --- |
| Cenário | Uma topologia por liga; nunca misturar snapshots entre topologias. |
| Bootstrap | Um snapshot de cada papel treinado contra os baselines v0.3. |
| Ordem | Atacante, posicionador, atacante, posicionador… |
| Liga | Uniforme entre snapshots do papel oposto, incluindo históricos. |
| Seleção | Seed de seleção e seed de treino persistidas por rodada. |
| Avaliação | Contra baselines congelados e snapshots de referência, além do oponente da rodada. |
| Promoção | Por métricas de validação; teste final separado e usado uma vez. |
| Parada | Rodadas, orçamento por papel e critério de falha definidos antes da execução. |

Cada `round-XXX.json` salva os targets congelados, as métricas retornadas e o
estado `promotion: not-decided`. Portanto, vencer o oponente corrente nunca
promove um modelo automaticamente: a decisão exige a análise multi-seed de
validação e, se aprovada, um único teste cego separado.

### Esqueleto de execução

```python
runner = CoupledSelfPlayRunner(
    record=campaign_record,
    topology=PERIODIC_TABLE_BATTLESHIP,
    trainer=experiment_trainer,
    frozen_suite=fixed_suite_evaluator,
    runtime_opponents=loaded_bootstrap_snapshots,
    output_directory=Path("runs/v0.5/self-play"),
)

while runner.run_next_round() is not None:
    pass
```

O `experiment_trainer` deve respeitar `plan.training_seed`, `plan.timesteps` e
o oponente congelado recebido. Para um atacante, ele recebe `CoupledAttackEnv`;
para um posicionador, recebe `DefensiveEvaluator`. Ao retornar, deve informar
o caminho local do checkpoint, o `source_run_id` e o adaptador carregado do
novo snapshot. O caminho é usado apenas para calcular o hash e não aparece no
ledger.

O primeiro piloto deve ter poucas rodadas e orçamento declarado. Ele valida o
fio completo, não procura recorde. Só uma campanha posterior, pré-registrada e
com a suíte congelada, pode gerar uma candidata para o placar do README.
