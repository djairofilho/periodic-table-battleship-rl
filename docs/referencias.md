# Referências

## Ambiente e treinamento

- [Gymnasium: criação de ambiente customizado](https://gymnasium.farama.org/main/tutorials/environment_creation/).
  Define o contrato de `Env`, espaços, `reset`, `step` e registro.
- [Gymnasium: action masking no Taxi](https://gymnasium.farama.org/tutorials/training_agents/action_masking_taxi/).
  Referência para restringir a escolha aos alvos válidos.
- [sb3-contrib: MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).
  Implementação planejada para o primeiro baseline de RL. A documentação pede
  que a máscara seja implementada no próprio ambiente quando houver processos
  paralelos e que a avaliação use as utilidades maskable.
- [uv: projetos Python](https://docs.astral.sh/uv/guides/projects/).
  Referência para `pyproject.toml`, grupos de dependência e `uv.lock`.

## Regras e química

- [Teachwire: Periodic Table Battleship](https://www.teachwire.net/teaching-resources/periodic-table-battleship-game-for-y6-y7-chemistry/).
  Referência didática para a frota `2, 3, 3, 4, 5`, orientações ortogonais e
  contato permitido entre navios.
- [Montejo Bernardo e Fernández González (2021), Chemical Battleship](https://doi.org/10.1021/acs.jchemed.0c00553).
  Referência acadêmica sobre uma Batalha Naval didática na tabela periódica.
- [IUPAC: Periodic Table of the Elements](https://iupac.org/what-we-do/periodic-table-of-elements/).
  Fonte de autoridade planejada para o catálogo de elementos.

## Projeto de origem

As regras foram analisadas localmente no projeto `periodic-table-battleship`,
commit `229d0ba`. Ele é uma referência de produto e não é dependência deste
repositório. Nenhum código, asset ou catálogo foi copiado.
