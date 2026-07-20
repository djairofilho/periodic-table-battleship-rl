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

Nesta etapa o repositório contém a especificação e a infraestrutura mínima do
projeto. O ambiente, os agentes treináveis e os resultados serão implementados
em etapas posteriores, para que as regras e o protocolo sejam definidos antes
de gerar métricas.

## Comparação inicial

| Cenário | Grade | Células jogáveis | Regra comum |
| --- | ---: | ---: | --- |
| `classic_10x10` | 10 × 10 | 100 | Frota `2, 3, 3, 4, 5` |
| `periodic_table_118` | 10 × 18 | 118 | Frota `2, 3, 3, 4, 5` |

Em ambos os casos, navios são lineares e ortogonais, não se sobrepõem e podem
encostar. No cenário periódico, uma célula corresponde a um elemento. As
lacunas da tabela não são alvos. O agente enfrenta uma frota adversária
amostrada de forma legal, sem acesso ao seu estado secreto.

O espaço de ação será comum: uma posição em uma tela de `10 × 18`. Uma máscara
de ação elimina lacunas, células fora da grade do cenário e tiros repetidos.
Isso mantém a API compatível e impede que escolhas impossíveis contaminem o
treinamento.

## Documentação

- [Análise do jogo de origem](docs/01-analise-do-jogo-origem.md)
- [Especificação do ambiente](docs/02-especificacao-do-ambiente.md)
- [Protocolo de benchmark](docs/03-protocolo-de-benchmark.md)
- [Roadmap](docs/04-roadmap.md)
- [Referências](docs/referencias.md)

## Desenvolvimento

O projeto usa [uv](https://docs.astral.sh/uv/) e Python 3.11.

```powershell
uv sync --all-groups
uv run ruff check .
uv run pytest
```

As dependências de treinamento pesado serão opcionais. O núcleo do ambiente
ficará leve, baseado em `gymnasium` e `numpy`.

## Escopo inicial

- Ambiente Gymnasium de agente único para a fase de tiros.
- Dois cenários comparáveis: grade clássica e tabela periódica.
- Baselines reproduzíveis: aleatório e hunt-target.
- Primeiro agente de RL: MaskablePPO.
- Avaliação por sementes fixas, métricas de eficiência e relatórios versionados.

O posicionamento por RL, o jogo competitivo entre dois agentes e recursos
educacionais da interface original são extensões futuras.

## Licença

Distribuído sob a [Licença MIT](LICENSE).
