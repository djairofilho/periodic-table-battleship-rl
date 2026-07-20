# PPO com CNN espacial

Issue: [#49](https://github.com/djairofilho/periodic-table-battleship-rl/issues/49)

## Hipótese

O controle `maskable-ppo-v1` achata o tabuleiro público antes de decidir. A
candidata `maskable-ppo-cnn-v1` recebe os mesmos planos públicos e a mesma
máscara de ação, mas usa convoluções para preservar adjacências. A hipótese é
que isso ajuda a transição de caça para alvo após um acerto.

## Contrato da candidata

- Não altera o ambiente, as recompensas nem o controle MLP da v0.4.
- O extrator usa duas convoluções `3×3` e *adaptive pooling* para um vetor de
  128 características. O plano de validade continua explícito, portanto as
  lacunas periódicas não viram células jogáveis.
- A política ainda só recebe observação Gym pública e `action_masks()`.
- Os metadados usam `attack-cnn-training-v1` e o id de política próprio. Não é
  permitido reportá-la como `maskable-ppo-v1`.

## Execução de piloto

```powershell
uv run --extra train python scripts/run_ppo_cnn_pilot.py --scenario battleship --smoke
uv run --extra train python scripts/run_ppo_cnn_pilot.py --scenario dense-118
```

Os checkpoints são comparados apenas nas cinco seeds de validação fixas. O
comando não aceita nem consulta a agenda de teste cego.

## Promoção e apresentação

Para promoção, a CNN precisa superar o MLP em validação multi-seed com IC 95%
favorável para `valid_shots` menor, e repetir a vantagem no teste cego. O
relatório da campanha deve incluir:

1. curva de validação por checkpoint;
2. tabela pareada CNN menos MLP e CNN menos hunt-target;
3. heatmap de tiros e de tiros após acerto;
4. GIF lado a lado na mesma seed de demonstração, fora do teste cego.

O README recebe somente a linha promovida no placar geral; gráficos e todos os
checkpoints ficam no relatório detalhado da campanha.
