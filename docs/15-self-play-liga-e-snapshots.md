# Self-play: liga, snapshots e avaliação estável

A issue [#40](https://github.com/djairofilho/periodic-table-battleship-rl/issues/40)
introduz a base reprodutível para self-play. Aqui, *self-play* significa treinar
alternadamente o atacante contra posicionadores congelados e o posicionador
contra atacantes congelados. Não é uma única política jogando contra si mesma:
cada papel aprende uma decisão diferente e a avaliação precisa continuar
comparável enquanto os oponentes evoluem.

## Contrato implementado

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

## Protocolo para o primeiro experimento acoplado

Ainda falta um ambiente acoplado que aceite um atacante e um posicionador PPO
congelados como oponentes. Quando ele existir, a campanha inicial deve usar:

| Campo | Decisão inicial |
| --- | --- |
| Cenário | uma topologia por liga; nunca misturar snapshots entre topologias |
| Bootstrap | um snapshot de cada papel treinado contra os baselines v0.3 |
| Ordem | atacante, posicionador, atacante, posicionador… |
| Liga | uniforme entre snapshots do papel oposto, incluindo históricos |
| Seleção | seed de seleção e seed de treino persistidas por rodada |
| Avaliação | contra os baselines congelados e snapshots de referência, além do oponente da rodada |
| Promoção | por métricas de validação; teste final separado e usado uma vez |
| Parada | número de rodadas, orçamento por papel e critério de falha definidos antes da execução |

Uma melhora contra o oponente corrente não basta: ela pode ser um ciclo entre
políticas. Só uma melhora repetida na suíte congelada, sem regressão acentuada
contra snapshots anteriores, pode justificar promoção. A infraestrutura atual
foi deliberadamente separada do ambiente para que esse experimento não altere
os benchmarks v0.3 nem transforme seus resultados em ajuste retrospectivo.
