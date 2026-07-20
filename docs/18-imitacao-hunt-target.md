# Imitação pública de hunt-target e ajuste por PPO

Issue: [#51](https://github.com/djairofilho/periodic-table-battleship-rl/issues/51)

## Hipótese

`hunt-target` já codifica a estratégia local que faltou ao PPO: depois de um
acerto, investigar os vizinhos ortogonais. O pré-treino por imitação fornece
esse comportamento inicial sem revelar a frota. O PPO posterior só pode
ajustar a política usando recompensa do ambiente.

## Dataset auditável

`generate_hunt_target_dataset` registra exatamente três arrays comprimidos:

| Campo | Origem pública |
| --- | --- |
| `observations` | observação Gym antes do tiro |
| `action_masks` | `AttackEnv.action_masks()` antes do tiro |
| `actions` | decisão legal de hunt-target |

O arquivo `dataset.json` enumera explicitamente os campos excluídos: frota,
células ocupadas, ids e posicionamentos dos navios. Os acertos ativos do
professor são reconstruídos somente do canal público de acertos ativos.

## Execução de piloto

```powershell
uv run --extra train python scripts/run_hunt_target_imitation_pilot.py --smoke
uv run --extra train python scripts/run_hunt_target_imitation_pilot.py --scenario dense-118
```

O piloto salva o clone comportamental, checkpoints do fine-tuning e hashes do
dataset. É proibido usar teste cego para escolher época, checkpoint ou
hiperparâmetro.

## Interpretação e promoção

O relatório deve separar três linhas: PPO-MPL controle, imitação pura e
imitação mais PPO. A seleção usa somente validação fixa. Para cada linha,
inclua curva de perda de imitação, curva de `valid_shots`, teste pareado,
heatmap e GIF de uma seed demonstrativa separada. Uma candidata só atualiza o
placar compacto do README após confirmar o ganho no teste cego.
