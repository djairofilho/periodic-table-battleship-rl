# Reprodutibilidade

## Ambiente de execuÃ§Ã£o

```powershell
uv sync --all-groups --extra visual --extra docs
uv run ruff check .
uv run pytest
```

## Campanhas em validaÃ§Ã£o

Cada campanha possui:

- parÃ¢metros explÃ­citos (`seed`, `episodes_per_seed`, `sample_count`),
- JSON de protocolo,
- saÃ­da em `runs/` e artefatos em `artifacts/`,
- resumo em Markdown.

## Comando de sincronizaÃ§Ã£o de ativos do site

```powershell
uv run python scripts/sync_site_assets.py --strict
```

O comando copia ativos grÃ¡ficos estÃ¡veis para `docs/assets` e escreve
`docs/assets/site-asset-manifest-v0.8.json` com checksums.

## Checklist antes de publicar

1. `mkdocs build` sem erros.
2. `git status` limpo para as evidÃªncias da release em questÃ£o.
3. `docs/assets/site-asset-manifest-v0.8.json` com hashes atualizados.
4. PÃ¡gina de resultados atualizada com decisÃ£o explÃ­cita.


