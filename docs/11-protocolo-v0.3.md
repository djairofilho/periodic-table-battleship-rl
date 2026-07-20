# Protocolo v0.3: campanha escalonada e controlada

Esta versão amplia a campanha piloto v0.2 sem mudar o princípio central:
treino, validação e teste são conjuntos de seeds separados; somente a
validação seleciona modelos ou hiperparâmetros.

## Decisões congeladas

| Tema | Decisão |
| --- | --- |
| Atacantes de posicionamento | Suíte fixa: aleatório, hunt-target, PPO congelado e mistura |
| Baselines de posicionamento | Random legal, dispersão e resistência a hunt-target |
| Dados tabulares | Pandas, usado apenas onde a visualização Seaborn precisa de DataFrame |
| Self-play | Adiado para a issue [#40](https://github.com/djairofilho/periodic-table-battleship-rl/issues/40) |
| Execução | Sequencial, com uma thread PyTorch para reduzir overhead no ambiente pequeno |

## Capacidade medida

O computador de referência é um Acer Nitro AN515-54 com i5-9300H, 32 GB de
RAM, GTX 1650 e PyTorch CPU-only. Uma medição local registrou 472 passos/s no
ataque e 360 passos/s no posicionamento, ambos com uma thread. A GPU não é
usada nesta versão porque a instalação atual do PyTorch não contém CUDA.

## Busca e campanha de ataque

| Fase | Cenários | Treino | Validação | Teste |
| --- | --- | --- | --- | --- |
| Busca | clássico, `dense-118`, periódico | 3 candidatos × 3 seeds × 20 mil passos | 10 seeds | nenhum |
| Final | os três | 5 seeds × 50 mil passos | 10 seeds, checkpoints em 10k/20k/30k/40k/50k | 100 seeds cegos |

Os candidatos variam taxa de aprendizado e tamanho de rollout. A seleção usa
menor média de `valid_shots`; empates usam `candidate_id` de modo determinístico.
O melhor checkpoint final por cenário é selecionado na validação e comparado
no teste contra `random_masked-v1` e `hunt_target-v1`.

## Campanha de posicionamento

| Fase | Cenários | Treino | Validação | Teste |
| --- | --- | --- | --- | --- |
| Final | clássico e periódico | 5 seeds × 50 mil passos | 10 seeds, contra a mistura congelada | 100 seeds por componente e mistura |

Antes do PPO, as três baselines independentes são avaliadas nos mesmos seeds,
atacantes e regras de desempate. O posicionador PPO é treinado contra a mistura
equiponderada de atacante aleatório, hunt-target e PPO de ataque congelado do
mesmo cenário.

## Métricas e gates

- Ataque: menor `valid_shots` é melhor.
- Posicionamento: maior `valid_shots_to_sink` é melhor.
- Comparações usam diferença pareada por seed e bootstrap percentil com 10 mil
  reamostragens.
- O controle `dense-118` separa, parcialmente, cardinalidade de células da
  geometria irregular.
- O teste só ocorre após a seleção ser persistida; não entra em tuning,
  escolha de seed nem leitura de curva.

## Artefatos esperados

O executor `scripts/run_v0_3_campaign.py` grava checkpoints locais em
`.local-runs/v0.3-fixed-suite` e resultados públicos em
`runs/v0.3-fixed-suite` e `artifacts/v0.3-fixed-suite`. A release v0.3 deve
incluir JSONL, manifests, tabelas, gráficos, heatmaps, curvas por checkpoint e
GIFs sem revelar frotas privadas.
