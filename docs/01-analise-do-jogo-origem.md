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

## Experimento de ataque

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

## Experimento de posicionamento

O posicionamento não será aprendido no mesmo ambiente do ataque nesta primeira
etapa. Ele terá outro ambiente: o agente posiciona os cinco navios em ordem
fixa, sempre escolhendo uma colocação legal mascarada. Depois da quinta escolha,
uma política atacante da suíte de avaliação joga até afundar a frota.

O retorno do posicionador será proporcional à sobrevivência: quanto mais tiros
válidos o atacante precisar para afundar os 17 segmentos, maior a recompensa.
O treino inicial ocorrerá contra uma mistura fixa de atacantes aleatório e
hunt-target. Isso evita que a política de posicionamento persiga um oponente
que muda a cada atualização.

## Decisões adiadas

- Restrições que impeçam navios adjacentes.
- Cartas táticas, `insight` e perguntas de química.
- Identificação de alvos por símbolo, grupo ou número atômico.
- Self-play entre atacante e posicionador.

Essas extensões alteram a distribuição de estados ou introduzem outra tarefa de
decisão. Elas só entram depois que os dois cenários básicos tiverem testes e
resultados de referência.
