# Demonstrações Bayesianas públicas para destilação v0.7

## Objetivo

O planejador `belief_probability_mc-v1` é uma referência forte, mas seu custo
por decisão cresce com o número de amostras de frotas compatíveis. Esta etapa
gera demonstrações para uma política neural imitar o professor sem receber
qualquer informação que um atacante humano não possuiria.

Cada registro é produzido antes do tiro e contém somente:

- `observations`: os quatro canais públicos do `AttackEnv`;
- `action_masks`: tiros ainda válidos;
- `teacher_actions`: ação legal escolhida pelo professor;
- `teacher_occupancy_probabilities`: escores de ocupação inferidos pelo
  professor, zerados nas ações indisponíveis.

O arquivo NPZ rejeita qualquer outro campo. Em especial, não há `fleet`,
`occupied_cells`, IDs, posições de navios, recompensas privadas ou rótulos de
acerto futuro. Os escores são estimativas Monte Carlo derivadas do histórico
público e não um posterior declarado como exato.

## Reprodutibilidade

A agenda de episódios, a seed interna do amostrador, os limites de
backtracking, a configuração do ambiente e o SHA-256 do NPZ ficam em
`dataset.json`. Reexecutar a mesma configuração substitui o mesmo artefato com
o mesmo conteúdo. O gerador usa `SeedSequence(episode_seed, sampler_seed)`;
logo, uma nova seed de episódio não altera as demonstrações já definidas.

O artefato de auditoria padrão usa apenas a seed de treino `9701`. Ela não é
uma seed de validação nem abre o inventário de teste cego.

```powershell
uv run python scripts/generate_bayesian_demonstrations.py
uv run pytest tests/training/test_bayesian_distillation.py
```

Para uma agenda maior, passe explicitamente seeds de treino e preserve o
manifesto resultante:

```powershell
uv run python scripts/generate_bayesian_demonstrations.py `
  --dataset-id v0.7-bayes-train-32 `
  --sample-count 32 `
  --sampler-seed 0 `
  --seeds 9701 9702 9703 9704
```

## Contrato para o treinador

`load_bayesian_demonstrations` valida tipo, forma, máscara, intervalo dos
escores e que a ação do professor é o argmax estável de seu escore público. O
treinador da issue D2 deve usar apenas esse carregador e registrar o SHA-256 do
conjunto consumido. Uma divergência de esquema ou um campo adicional falha
antes do treino.
