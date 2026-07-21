# Periodic Table Battleship RL

[![CI](https://img.shields.io/github/actions/workflow/status/djairofilho/periodic-table-battleship-rl/ci.yml?branch=main&label=CI)](https://github.com/djairofilho/periodic-table-battleship-rl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/djairofilho/periodic-table-battleship-rl/docs.yml?branch=main&label=Docs)](https://github.com/djairofilho/periodic-table-battleship-rl/actions/workflows/docs.yml)
[![Docs do Projeto](https://img.shields.io/badge/docs-Periodic%20Table%20Battleship%20RL-0A66C2)](https://djairofilho.github.io/periodic-table-battleship-rl/)
[![License MIT](https://img.shields.io/github/license/djairofilho/periodic-table-battleship-rl)](LICENSE)

Projeto de *reinforcement learning* para Batalha Naval em duas grades:
`battleship` (10×10) e `periodic-table-battleship` (10×18).

## Em 1 minuto

- Ambiente Gymnasium com observações públicas, ação mascarada e contratos auditáveis;
- Cenários: `battleship`, `periodic-table-battleship`, `dense-118`;
- Testes por split (treino/validação/teste cego) e rastreabilidade por artefato;
- Baselines, política Bayesiana e distilação pública na trilha v0.9.

## Estado atual

- Baseline `hunt-target-v1` segue como referência principal.
- Planejador `belief_probability_mc-v1` melhorou no smoke multi-topologia, sem campanha de validação ampla para promoção.
- CNN/GNN foram testadas em multi-seed e não promoveram em v0.9.
- Self-play permanece em v0.10, após nova candidata robusta.

A decisão final da versão v0.9 está em:

- [Relatório v0.9](docs/32-relatorio-v0.9.md)

## Documentação científica (resumo)

Use o site para leitura completa: 

- [Visão geral](docs/visao-geral.md)
- [Especificação do jogo](docs/jogo.md)
- [Métodos de ataque](docs/metodos-ataque.md)
- [Resultados](docs/resultados.md)
- [Reprodutibilidade](docs/reproducibilidade.md)
- [Roadmap 0.8–0.9](docs/roadmap-0.8-0.9.md)
- [Galeria de gráficos e GIFs](docs/galeria.md)

## Como executar (resumo)

```powershell
uv sync --all-groups --extra visual --extra docs
uv run ruff check .
uv run pytest
uv run python scripts/sync_site_assets.py --strict
uv run mkdocs build
```

## Repositório e licença

- MIT: [LICENSE](LICENSE)
- GitHub: [https://github.com/djairofilho/periodic-table-battleship-rl](https://github.com/djairofilho/periodic-table-battleship-rl)
