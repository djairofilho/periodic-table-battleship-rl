# DQN mascarado para ataque

Esta implementação introduz `masked-dqn-v1` como comparação *off-policy* ao
MaskablePPO. Ela usa replay buffer, rede-alvo e uma MLP pequena em PyTorch.
Não altera o ambiente, a frota aleatória, a observação pública nem a máscara de
ações do benchmark.

## Segurança da máscara

A máscara é aplicada em dois pontos que precisam permanecer separados:

1. Na política de comportamento, exploração escolhe somente índices válidos e
   exploração gulosa faz `argmax` depois de mascarar índices ilegais.
2. No alvo de Bellman, o máximo da rede-alvo também ignora ações ilegais do
   próximo estado. Estados sem ação seguinte legal recebem bootstrap zero.

Assim uma lacuna da tabela periódica, uma célula já chamada ou um estado
terminal não pode inflar artificialmente o valor de uma ação. As transições de
replay guardam a máscara do próximo estado, sem expor a ocupação da frota.

## Piloto reproduzível

O runner abaixo treina uma única seed pequena e a avalia em seeds de validação
separadas. Ele gera pesos em `.local-runs` e episódios/manifests públicos em
`runs`; é uma verificação de integração, não uma campanha comparável.

```powershell
uv run --extra train python scripts/run_masked_dqn.py
```

Para mudar o orçamento do piloto:

```powershell
uv run --extra train python scripts/run_masked_dqn.py --steps 10000 --test-seeds 20
```

Uma candidata só pode entrar no placar após definir seeds de treino, validação
e teste cegos, selecionar checkpoint apenas na validação e superar o controle
com intervalo pareado de 95%. O piloto traz `promotion_eligible: false` no
manifest exatamente para impedir interpretação indevida.

## Artefatos

O checkpoint `model.pt` contém pesos e dimensões da rede. `training.json`
registra topologia, configuração do ambiente, hiperparâmetros e versão do
PyTorch. A avaliação registra hashes do checkpoint e do metadata, hash de
`uv.lock`, commit Git, seeds e protocolo `blind-public-observation-masked-dqn-v1`.
