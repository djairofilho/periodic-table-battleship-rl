# Análise do jogo de origem

## Recorte analisado

O projeto de origem, `periodic-table-battleship`, modela uma Batalha Naval
digital em que cada elemento é uma célula jogável. Esta análise usa o estado
do repositório em `229d0ba` apenas como referência de regra e de topologia. Não
há cópia de código, assets ou dados desse projeto para este repositório.

## Regras preservadas no benchmark inicial

| Aspecto | Regra observada | Decisão para o benchmark |
| --- | --- | --- |
| Células | Um alvo válido representa exatamente um elemento. | Preservar. Há 118 alvos no cenário periódico. |
| Geometria | A tabela tem lacunas e duas séries inferiores. | Preservar em uma malha `10 × 18`. |
| Frota | Tamanhos `2, 3, 3, 4, 5`. | Preservar nos dois cenários. |
| Posicionamento | Linear, horizontal ou vertical, sem sobreposição. | Preservar. |
| Contato | Permitido, pois a regra atual só impede sobreposição. | Preservar para equivalência. |
| Tiro | Só pode atingir uma célula válida e ainda não chamada. | Aplicar com máscara de ação. |
| Informação | O oponente não vê a frota secreta. | Preservar estritamente na observação. |

O total da frota é 17. Uma partida termina quando os 17 segmentos foram
atingidos.

## O que muda para reinforcement learning

O jogo de interface alterna turnos entre uma pessoa e a CPU. Para isolar a
decisão de busca, a primeira versão do ambiente é de agente único: em cada
episódio, uma frota inimiga legal é amostrada e permanece oculta; o agente
escolhe tiros até afundá-la. Não haverá turno reativo do oponente nessa etapa.

Isso permite medir a eficiência de encontrar uma frota sob informação parcial
sem misturar, no mesmo experimento, uma política de tiro e uma política de
posicionamento.

## Risco principal de comparação

A tabela periódica tem 118 posições válidas, contra 100 no tabuleiro clássico,
e a distribuição dessas posições não é retangular. Comparar somente a taxa de
vitória seria inútil, pois um agente que nunca repete tiros eventualmente vence
nos dois cenários. O protocolo deve reportar principalmente quantos tiros
válidos foram necessários e a curva de segmentos encontrados ao longo da
partida.

## Decisões adiadas

- Restrições que impeçam navios adjacentes.
- Cartas táticas, `insight` e perguntas de química.
- Identificação de alvos por símbolo, grupo ou número atômico.
- Treinamento de posicionamento e self-play.

Essas extensões alteram a distribuição de estados ou introduzem outra tarefa de
decisão. Elas só entram depois que os dois cenários básicos tiverem testes e
resultados de referência.
