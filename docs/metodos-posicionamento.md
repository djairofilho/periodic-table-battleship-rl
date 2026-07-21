# Métodos: posicionamento

## Método de posicionamento

O posicionamento é tratado como agente defensivo com política de colocação legal:

1. gerar uma frota legal para cada episódio;
2. simular ataques com alvo de treino;
3. selecionar ações com base no objetivo de atraso de descoberta.

## Métricas de referência

- mais `valid_shots` do atacante até o fim = posição mais robusta;
- distribuição espacial dos navios final para inspecionar concentração e gaps.

## Estado atual

A integração com o ciclo de avaliação de posicionamento segue a mesma regra de separação:

- validação com atacantes fixos;
- promoção somente com ganhos estáveis;
- self-play condicional apenas no próximo bloco.

