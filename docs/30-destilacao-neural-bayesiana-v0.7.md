# Destilação neural Bayesiana v0.7

## Objetivo

Esta etapa testa se uma rede pública, barata em inferência, consegue imitar o
planejador Bayesiano de maior probabilidade. Ela não é uma campanha de
reinforcement learning e não consome o conjunto de teste cego.

O professor produz, a cada estado público, uma estimativa Monte Carlo de
ocupação para as ações legais. O estudante recebe somente:

- os quatro planos públicos da observação de ataque;
- a máscara de ações legais;
- a ação escolhida e o mapa público de probabilidades do professor.

Frota, ocupação real, identidade de navios, recompensa privada e estado
interno do ambiente são excluídos do arquivo `.npz` e validados pelo schema
`bayesian-public-demonstrations-v1`.

## Modelos e perda

Cada topologia possui seu próprio estudante, pois o GNN incorpora a adjacência
legal da topologia. A CNN usa convoluções 3×3 no canvas de 10×18; o GNN usa
duas etapas de passagem de mensagens somente entre células válidas.

Para logits mascarados `z`, ação do professor `a` e probabilidades públicas
normalizadas `p`, a função de perda é:

```math
L = (1 - \lambda)\,\operatorname{CE}(z, a)
    + \lambda\,\operatorname{KL}(p \parallel \operatorname{softmax}(z)),
\qquad \lambda = 0{,}35.
```

As ações inválidas recebem logit finito muito negativo durante a perda. Isso
preserva a máscara e evita o `0 × -∞` que tornaria a divergência KL indefinida.
Na decisão, a máscara é aplicada novamente de modo estrito.

## Protocolo do piloto

O programa [run_bayesian_student_pilot.py](../scripts/run_bayesian_student_pilot.py)
usa seeds de treino `9601–9602` e seeds distintas de validação `9651–9652`.
Ele gera datasets públicos separados, treina CNN e GNN em cada uma das três
topologias e mede:

- acordo de ação contra as demonstrações do professor fora do treino;
- tiros válidos e AUC de descoberta nas duas partidas de validação;
- tiros válidos do `hunt-target-v1` nas mesmas frotas.

```powershell
uv run --extra visual python scripts/run_bayesian_student_pilot.py --smoke
```

O modo padrão aumenta a amostra Monte Carlo e os epochs, mas permanece uma
validação de seleção. O script não possui argumento de `split` e escreve
explicitamente `blind_test_used: false`.

## Resultado inicial e decisão

O piloto mínimo foi executado com duas amostras Monte Carlo e quatro epochs.
O GNN teve resultado pontual melhor que `hunt-target` apenas em Batalha Naval
clássica (46,00 vs. 54,00 tiros), mas CNN e GNN ficaram piores nas duas
topologias de 118 células. O acordo fora do treino foi baixo (1,7% a 11,4%).

Portanto, nenhum estudante é promovido. Além do desempenho inconsistente, duas
seeds são insuficientes para uma decisão de promoção: a promoção exige a
validação multi-seed pré-registrada e um intervalo pareado de 95% favorável em
todas as topologias. O resultado é útil porque torna a hipótese falsificável:
antes de escalar na GPU, é preciso aumentar o dataset público, calibrar o
professor e demonstrar generalização em validação.

Os artefatos reproduzíveis estão em
[`artifacts/v0.7-bayesian-students`](../artifacts/v0.7-bayesian-students/):

- `student-pilot-report.json`: métricas, seeds e hashes dos datasets;
- `student-pilot-summary.md`: tabela curta;
- `student-valid-shots.png`: gráfico comparativo;
- `datasets/*/dataset.json`: certificados de dados públicos.
