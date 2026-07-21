# Especificação do jogo e contratos

## Contrato principal do ambiente

- Estado público é a observação do jogo sem acesso à frota real.
- Espaço de ação é fixo em `10 × 18` (quando aplicável), com máscara legal.
- Não há acções inválidas em treino: elas recebem máscara `False`.

## Observações da API

- Ambiente de ataque:
  - `reset(seed)` inicia estado íntegro e reprodutível.
  - `step(action)` retorna observação com canais públicos, `terminated`, `truncated`
    e `info` com `valid_shots`.
- Ambiente de posicionamento:
  - política gera uma frota em cima da topologia e participa de confronto por mistura
    de atacantes defensivos.

## Contratos de dados públicos

Todo experimento preserva:

- inventário de seeds;
- parâmetros de campanha;
- commit e hash de `uv.lock`;
- artefatos de saída (`json`, `csv`, imagem, gif).

Isso evita ambiguidade entre validação e teste cego.

## Limitações explícitas

1. A seleção de frota do adversário é sempre legal e amostrada de forma reproduzível.
2. Estratégias de planejamento podem ser aproximadas (MC) e não exatas.
3. Mudanças de estratégia só entram em cego após gate pré-definido.

