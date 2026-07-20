# Protocolo v0.5: candidatas comparáveis e promoção auditável

Este contrato governa as experiências que sucedem a v0.4: PPO-CNN, DQN
mascarado, imitação mais PPO e GNN. Ele não altera nem reinterpreta os
resultados v0.3/v0.4.

## Regra central

Uma candidata é escolhida exclusivamente na validação. O teste cego confirma
uma escolha que já foi persistida; ele nunca escolhe arquitetura, recompensa,
checkpoint, seed ou hiperparâmetro.

```text
inventário congelado → treino → validação e registro de seleção
                                      ↓
                          somente então: teste cego
                                      ↓
                               promover ou rejeitar
```

## Contrato persistível

O módulo `periodic_table_battleship_rl.protocol.v05` fornece registros
imutáveis e serializáveis em JSON para cada experimento:

| Registro | Conteúdo obrigatório | Proteção oferecida |
| --- | --- | --- |
| `SeedInventory` | Seeds de treino, validação, teste e demonstração | Os quatro conjuntos são não vazios e mutuamente exclusivos. |
| `CheckpointPlan` | Passos, métrica, direção e split de seleção | O split de seleção é sempre `validation`. |
| `CandidateRegistration` | Controle, candidata, checkpoint e seeds de seleção | As seeds devem ser subconjunto da validação. |
| `TestConfirmation` | Candidata, id do registro e inventário de teste | Exige todas, e somente, as seeds fixas do teste. |
| `PromotionDecision` | Promoção ou rejeição e razão | Só existe com uma confirmação cega vinculada. |
| `ArtifactRecord` | Caminho relativo, hash e proveniência | Todo artefato declara a execução que o gerou. |

`ArtifactProvenance` inclui `run_id`, commit Git, hash de `uv.lock`, hash da
configuração e descrição do hardware. Assim um gráfico, GIF, CSV ou relatório
não depende de informação implícita da máquina que o produziu.

## Inventários obrigatórios

| Split | Uso permitido | Uso proibido |
| --- | --- | --- |
| `train` | Gradientes, replay buffer, ajuste interno da política | Comparação final publicada. |
| `validation` | Selecionar checkpoint e candidata; decidir se há hipótese para teste | Relatar como resultado final. |
| `test` | Uma confirmação pareada da candidata já registrada | Escolher qualquer detalhe da candidata. |
| `demonstration` | GIFs, replays e capturas de tela | Evidência estatística ou seleção. |

Os GIFs usam seeds de demonstração separadas para evitar que uma narrativa
visual seja escolhida procurando episódios do teste cego.

## Gate de promoção

Uma candidata entra no placar principal somente se:

1. tiver contrato, configuração e checkpoints pré-registrados;
2. vencer o controle na validação multi-seed pelo critério declarado;
3. tiver a seleção gravada em `CandidateRegistration` antes do teste;
4. mantiver a direção do efeito no teste cego pareado, com intervalo de
   confiança de 95% que satisfaça o limiar registrado;
5. publicar artefatos de resultado e proveniência completos.

Caso qualquer item falhe, a decisão é `rejected`; o relatório continua público
como evidência negativa e não substitui o controle no README.

## Uso mínimo

O executor cria o contrato antes de iniciar a confirmação cega e o grava junto
do manifesto da execução. A implementação deve testar o contrato com:

```powershell
uv run pytest tests/protocol/test_v05.py
```

O teste garante que uma seed de teste não pode entrar na seleção e que toda
confirmação aponta para uma seleção já identificada.
