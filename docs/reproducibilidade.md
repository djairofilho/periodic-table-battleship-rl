# Reprodutibilidade

## Ambiente de execução

```powershell
uv sync --all-groups --extra visual --extra docs
uv run ruff check .
uv run pytest
```

## Campanhas da v0.9

Cada campanha possui:

- parâmetros explícitos (`seed`, `episodes_per_seed`, `sample_count`);
- JSON de protocolo;
- saída em `runs/` e artefatos em `artifacts/`;
- relatório humano em Markdown.

## Comando de sincronização do site

```powershell
uv run python scripts/sync_site_assets.py --strict
```

O comando copia ativos gráficos estáveis para `docs/assets` e escreve
`docs/assets/site-asset-manifest-v0.8.json` com checksums (inclui entradas v0.9).

## Checklist antes de publicar

1. `mkdocs build` sem erros.
2. `git status` limpo para as evidências da release em questão.
3. `docs/assets/site-asset-manifest-v0.8.json` atualizado.
4. Página de resultados com decisão explícita e links de evidência.
