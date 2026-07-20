# Periodic Table Battleship RL

[![CI](https://img.shields.io/github/actions/workflow/status/djairofilho/periodic-table-battleship-rl/ci.yml?branch=main&label=CI)](https://github.com/djairofilho/periodic-table-battleship-rl/actions/workflows/ci.yml)
[![Licença MIT](https://img.shields.io/github/license/djairofilho/periodic-table-battleship-rl)](LICENSE)
[![Último commit](https://img.shields.io/github/last-commit/djairofilho/periodic-table-battleship-rl)](https://github.com/djairofilho/periodic-table-battleship-rl/commits/main)

Ambiente [Gymnasium](https://gymnasium.farama.org/) e protocolo de benchmark
para treinar agentes de reinforcement learning em uma Batalha Naval cuja grade
é a tabela periódica.

O projeto começa por uma comparação controlada com a Batalha Naval
tradicional. A pergunta é simples: como uma topologia irregular, com lacunas e
118 células válidas, muda o aprendizado de uma política de busca?

## Estado

O núcleo reproduzível está disponível: topologias, frotas legais, ambientes
Gymnasium mascarados de ataque e posicionamento, baselines, persistência de
resultados, renderização pública de episódios e o microambiente tabular de
Q-learning/SARSA. Os pipelines MaskablePPO, avaliações cegas, tabelas,
gráficos e GIFs também estão disponíveis. A próxima etapa é ampliar os
orçamentos de treino e executar o protocolo final em múltiplas seeds.

## Cenários

| Cenário | Grade | Células jogáveis | Regra comum |
| --- | ---: | ---: | --- |
| `battleship` | 10 × 10 | 100 | Frota `2, 3, 3, 4, 5` |
| `periodic-table-battleship` | 10 × 18 | 118 | Frota `2, 3, 3, 4, 5` |

Em ambos os casos, navios são lineares e ortogonais, não se sobrepõem e podem
encostar. No cenário periódico, uma célula corresponde a um elemento. As
lacunas da tabela não são alvos. O agente enfrenta uma frota adversária
amostrada de forma legal, sem acesso ao seu estado secreto.

O espaço de ação será comum: uma posição em uma tela de `10 × 18`. Uma máscara
de ação elimina lacunas, células fora da grade do cenário e tiros repetidos.
Isso mantém a API compatível e impede que escolhas impossíveis contaminem o
treinamento.

## Experimentos

1. **Ataque:** a frota adversária é posicionada por uma política aleatória
   legal. O agente aprende onde atirar a partir de acertos e erros anteriores.
2. **Posicionamento:** um agente posiciona a própria frota e tenta maximizar o
   número de tiros necessários para que atacantes de uma suíte fixa a encontrem.

Os dois experimentos são independentes na primeira rodada. O segundo começa
contra atacantes fixos para que a recompensa seja estável. Self-play só entra
depois, como extensão.

## Benchmark inicial dos baselines

O primeiro benchmark reproduzível usa 20 seeds (`1001` a `1020`) e cinco
episódios por seed em cada combinação. Os dados brutos, manifesto e resumo
estão em [`runs/initial-baselines-v0`](runs/initial-baselines-v0).

| Cenário | Política | Episódios | Média de tiros válidos |
| --- | --- | ---: | ---: |
| `battleship` | `random_masked-v1` | 100 | 95,57 |
| `battleship` | `hunt_target-v1` | 100 | 59,28 |
| `periodic-table-battleship` | `random_masked-v1` | 100 | 112,72 |
| `periodic-table-battleship` | `hunt_target-v1` | 100 | 70,00 |

Estes números são uma linha de base, não um resultado de treinamento. Cada
manifesto registra o commit, o hash de `uv.lock`, seeds e ambiente de execução.

## Artefatos visuais e smoke runs PPO

Os dados iniciais dos baselines já têm [CSV e tabela](artifacts/initial-baselines-v0/tables),
[gráfico comparativo](artifacts/initial-baselines-v0/figures/mean-valid-shots.png)
e [GIF público de ataque](artifacts/initial-baselines-v0/gifs/hunt-target-demo.gif).

Também há demonstrações ponta a ponta de PPO para
[ataque](runs/attack-ppo-smoke-v0) e
[posicionamento](runs/placement-ppo-smoke-v0), além de
[gráficos, heatmap e GIF](artifacts/placement-ppo-smoke-v0). Elas usam somente
512 passos de treino e servem para verificar o pipeline. Não devem ser
interpretadas como avaliação final de desempenho.

## Documentação

- [Análise do jogo de origem](docs/01-analise-do-jogo-origem.md)
- [Especificação do ambiente](docs/02-especificacao-do-ambiente.md)
- [Protocolo de benchmark](docs/03-protocolo-de-benchmark.md)
- [Roadmap](docs/04-roadmap.md)
- [Experimentos e visualizações](docs/05-experimentos-e-visualizacoes.md)
- [Execução, Issues e trabalho paralelo](docs/06-execucao-e-rastreamento.md)
- [Contratos e critérios de aceite](docs/07-contratos-e-criterios-de-aceite.md)
- [Relatório v0.1](docs/08-relatorio-v0.1.md)
- [Referências](docs/referencias.md)

## Desenvolvimento

O projeto usa [uv](https://docs.astral.sh/uv/) e Python 3.11.

```powershell
uv sync --all-groups --extra visual
uv run ruff check .
uv run pytest
```

Para treinar PPO, acrescente `--extra train`. As dependências de visualização e
treino continuam opcionais; o núcleo do ambiente fica leve, baseado em
`gymnasium` e `numpy`.

## Escopo inicial

- Dois experimentos: ataque e posicionamento de frota.
- Baselines reproduzíveis: aleatório e hunt-target.
- Q-learning e SARSA em tabuleiros reduzidos; MaskablePPO nos cenários reais.
- Tabelas, gráficos Seaborn e GIFs determinísticos de partidas avaliadas.
- Avaliação por sementes fixas, métricas de eficiência e relatórios versionados.

O posicionamento por RL, o jogo competitivo entre dois agentes e recursos
educacionais da interface original são extensões futuras.

## Licença

Distribuído sob a [Licença MIT](LICENSE).
