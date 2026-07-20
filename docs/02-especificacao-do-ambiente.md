# Especificação do ambiente

## Objetivo

Disponibilizar um ambiente Gymnasium de agente único para aprender a escolher
tiros sobre uma frota inimiga oculta. A implementação seguirá o contrato
`reset() -> (observation, info)` e
`step(action) -> (observation, reward, terminated, truncated, info)`.

## Cenários

| Identificador | Células válidas | Topologia |
| --- | ---: | --- |
| `battleship` | 100 | Retângulo de 10 linhas por 10 colunas. |
| `periodic-table-battleship` | 118 | Tabela periódica com linhas principais e séries f separadas. |

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

Formato inicial: `Box(low=0, high=1, shape=(4, 10, 18), dtype=uint8)`.

| Canal | Conteúdo |
| --- | --- |
| 0 | Máscara fixa de células que pertencem ao cenário. |
| 1 | Acertos em navios ainda não afundados. |
| 2 | Acertos que pertencem a navios afundados. |
| 3 | Tiros que erraram. |

Quando um navio afunda, todos os seus segmentos mudam do canal 1 para o canal
2. O evento público também informa `sunk_ship_length` em `info`. Isso permite
que políticas e o baseline hunt-target saibam quais acertos ainda merecem busca
local, sem expor qualquer segmento não atingido.

A observação não contém a frota oculta, identidade química, número atômico ou
um contador de segmentos inimigos restantes. O objetivo inicial é isolar a
geometria e a cardinalidade do tabuleiro. Uma variante com atributos químicos
observáveis será outro cenário, não uma troca silenciosa de observação.

## Dinâmica

1. `reset(seed=...)` amostra uma frota inimiga legal com o gerador aleatório do
   ambiente e zera os tiros.
2. O agente escolhe uma célula ainda desconhecida.
3. O ambiente devolve acerto ou erro e atualiza a observação.
4. `terminated=True` quando todos os 17 segmentos forem atingidos.
5. Em uma trajetória válida, há no máximo uma tentativa por célula e a partida
   termina por vitória em no máximo `N` tiros válidos, onde `N` é 100 ou 118.
6. Ações fora da máscara existem somente para robustez e para `check_env`;
   recebem penalidade, não revelam informação e contam para um limite separado
   de tentativas totais. Alcançá-lo produz `truncated=True`.

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

## Política de posicionamento aleatório

No experimento de ataque, a frota será criada por `random_legal`: cada navio é
amostrado entre as colocações legais disponíveis naquele momento. Essa é uma
política aleatória de posicionamento, mas não deve ser chamada de uniforme
sobre todas as frotas completas, pois a construção sequencial pode favorecer
algumas configurações. A política, a ordem dos navios e as seeds serão
versionadas, para que o agente aprenda e seja avaliado contra a mesma
distribuição.

## Registro Gymnasium

Os identificadores planejados são `PTBRLBattleship-v0` e
`PTBRLPeriodicTableBattleship-v0`. A versão do sufixo só muda se o contrato de
observação, ação ou dinâmica for incompatível.

Os detalhes formais de topologia, índices, limites e critérios de aceite estão
em [Contratos e critérios de aceite](07-contratos-e-criterios-de-aceite.md).
