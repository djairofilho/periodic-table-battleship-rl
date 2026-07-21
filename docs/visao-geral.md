# Visao geral

## O que fazemos

Comparar estratégias de decisão em duas frentes:

- **Ataque:** escolher tiros otimos em um tabuleiro parcialmente oculto.
- **Posicionamento:** escolher uma frota inicial mais dificil de ser descoberta.

O ponto central é a comparação controlada: mesma API, mesmas sementes por
classe de experimento, mesmas regras de treino e um protocolo de promocao explicito.

## Cenarios

| Cenario | Grade | Celulas validas | Frota |
| --- | ---: | ---: | --- |
| `battleship` | 10×10 | 100 | 2, 3, 3, 4, 5 |
| `periodic-table-battleship` | 10×18 | 118 | 2, 3, 3, 4, 5 |
| `dense-118` | 10×18 | 118 | 2, 3, 3, 4, 5 |

## Modelo de validacao

1. Protocolo e contratos versionados em docs e codigo (`docs/26-protocolo-v0.7.md`).
2. Execução por sementes pre-registradas.
3. Métricas de tiro em validacao e testes cegos apenas quando o gate é aprovado.
4. Evidência auditavel para cada release (JSON, CSV, figuras, manifests).

## Decisao atual

- Se uma politica nao melhora em pelo menos dois cenarios com margem estatistica definida, nao avanca para teste cego.
- Resultados negativos sao documentados com a mesma prioridade dos positivos.
- O proximo ciclo nasce com hipotese nova e protocolo novo, nao com ajuste de hiperparametro isolado.
