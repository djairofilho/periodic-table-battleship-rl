# Especificação do ambiente

## Objetivo

Disponibilizar um ambiente Gymnasium de agente único para aprender a escolher
tiros sobre uma frota inimiga oculta. A implementação seguirá o contrato
`reset() -> (observation, info)` e
`step(action) -> (observation, reward, terminated, truncated, info)`.

## Cenários

| Identificador | Células válidas | Topologia |
| --- | ---: | --- |
| `classic_10x10` | 100 | Retângulo de 10 linhas por 10 colunas. |
| `periodic_table_118` | 118 | Tabela periódica com linhas principais e séries f separadas. |

Os dois cenários usarão uma tela lógica de 10 linhas por 18 colunas. No modo
clássico, apenas as 100 primeiras posições da grade 10 × 10 são válidas. No
modo periódico, as 118 posições dos elementos são válidas; lacunas continuam
inativas. A tabela de mapeamento entre posição, elemento, período, grupo e
série será versionada com fonte IUPAC antes de entrar no código.

## Ação e máscara

- `action_space`: `Discrete(180)`.
- Ação `a`: dispara na célula de índice de linha principal
  `linha * 18 + coluna`.
- `action_masks()`: vetor booleano de 180 posições. Uma ação é válida somente
  se a célula pertencer ao cenário e ainda não tiver sido chamada.

Uma ação dentro de `Discrete(180)` mas fora da máscara é tratada como no-op,
recebe a penalidade de ação inválida e consome uma tentativa. Essa escolha
mantém o ambiente robusto para verificadores que amostram ações sem máscara. Os
agentes avaliados pelo benchmark devem sempre usar a máscara.

## Observação

Formato inicial: `Box(low=0, high=1, shape=(3, 10, 18), dtype=uint8)`.

| Canal | Conteúdo |
| --- | --- |
| 0 | Máscara fixa de células que pertencem ao cenário. |
| 1 | Tiros que acertaram. |
| 2 | Tiros que erraram. |

A observação não contém a frota oculta, identidade química, número atômico ou
um contador de navios. O objetivo inicial é isolar o impacto da geometria. Uma
variante com atributos químicos observáveis será outro cenário, não uma troca
silenciosa de observação.

## Dinâmica

1. `reset(seed=...)` amostra uma frota inimiga legal com o gerador aleatório do
   ambiente e zera os tiros.
2. O agente escolhe uma célula ainda desconhecida.
3. O ambiente devolve acerto ou erro e atualiza a observação.
4. `terminated=True` quando todos os 17 segmentos forem atingidos.
5. `truncated=True` ao atingir o orçamento de tentativas da configuração.

O gerador de frotas precisa ser testado separadamente: cada navio deve ocupar
células contíguas da topologia do cenário, em uma única orientação, sem
sobreposição.

## Recompensa inicial

O perfil `efficiency-v0` será congelado antes do primeiro treinamento:

| Evento | Recompensa |
| --- | ---: |
| Acerto válido | `+1` |
| Erro válido | `-1` |
| Vitória | `+17` |
| Ação inválida | `-1` |

A métrica principal não é a recompensa: é o número de tiros válidos até
afundar a frota. A recompensa serve ao treinamento; as métricas servem à
comparação científica.

## Registro Gymnasium

Os identificadores planejados são `PeriodicBattleshipClassic-v0` e
`PeriodicBattleshipTable-v0`. A versão do sufixo só muda se o contrato de
observação, ação ou dinâmica for incompatível.
