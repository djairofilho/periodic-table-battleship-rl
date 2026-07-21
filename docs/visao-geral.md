# Visão geral

## O que fazemos

Comparar estratégias de decisão em duas frentes:

- **Ataque:** escolher tiros ótimos em um tabuleiro parcialmente oculto.
- **Posicionamento:** escolher uma frota inicial mais difícil de ser descoberta.

O ponto central é a *comparação controlada*: mesma API, mesmas sementes por
classe de experimento, mesmas regras de treino e um protocolo de promoção explícito.

## Cenários

| Cenário | Grade | Células válidas | Frota |
| --- | ---: | ---: | --- |
| `battleship` | 10×10 | 100 | 2, 3, 3, 4, 5 |
| `periodic-table-battleship` | 10×18 | 118 | 2, 3, 3, 4, 5 |
| `dense-118` | 10×18 | 118 | 2, 3, 3, 4, 5 |

## Modelo de validação

1. Protocolo e contratos versionados em docs e códigos (`docs/26-protocolo-v0.7.md`,
   `src/periodic_table_battleship_rl/evaluation`).
2. Execução em sementes pré-registradas.
3. Métricas de tiro em validação e testes cegos apenas quando a promoção foi
   tecnicamente desbloqueada.
4. Evidência auditável para cada release (JSON, CSV, figuras, manifests).

## Decisão atual

O projeto fecha ciclos de entrega em modo incremental:

- `<=` Se uma política não melhora em **pelo menos dois cenários** com margem de
  confiança definida, ela não avança para teste cego.
- `<=` Resultados negativos são documentados com a mesma prioridade dos positivos.
- `<=` O próximo ciclo nasce com hipótese nova e protocolo novo, não com "ajuste
  de hiperparâmetro isolado".

