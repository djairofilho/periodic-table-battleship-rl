# Periodic Table Battleship RL

## Resumo em 60 segundos

- **Objetivo:** treinar agentes de reinforcement learning para Batalha Naval nos cenarios:
  `battleship`, `periodic-table-battleship` e `dense-118`.
- **Contribuicao atual:** `hunt-target-v1` segue como baseline forte e a trilha v0.9 foi concluida com decisao negativa de promocao.
- **Estado cientifico:** os experimentos Bayesianos mostraram ganho em v0.9 smoke, mas nao preencheram criterio de gate para troca de politica.

## Comece aqui

1. [Visao geral e decisoes atuais](visao-geral.md)
2. [Especificacao do jogo e cenarios](jogo.md)
3. [Metodos de ataque](metodos-ataque.md)
4. [Resultados e interpretacoes](resultados.md)
5. [Reprodutibilidade](reproducibilidade.md)
6. [Roadmap tecnico](roadmap-0.8-0.9.md)

## Resultado mais recente (v0.9)

| Metrica | Melhor valor | Referencia | Decisao |
| --- | ---: | ---: | --- |
| Planejador Bayesiano (ataque) | 44.5 tiros em `periodic-table-battleship` | 64.0 (`hunt-target-v1`) | Melhor em par de cenarios no smoke, sem aprovacao para producao |
| CNN/GNN destiladas (ataque) | 55.25 tiros em `battleship` | 63.75 (`hunt-target-v1`) | Melhor em uma semente no multi-seed, nao melhora geral |
| Self-play de posicionamento | Nao iniciado | - | Adiado para v0.10 |

## Artefatos graficos principais

![Validação Bayesiana por topologia](assets/paired-valid-shots-v0.9.png)

![Calibracao do amostrador no microtabuleiro](assets/belief-sampler-calibration-v0.9.png)
