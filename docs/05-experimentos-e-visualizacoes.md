# Experimentos e visualizações

## Visão geral

O projeto tem dois problemas de aprendizado diferentes. Separá-los evita que o
resultado de uma política de ataque esconda o efeito de uma política de
posicionamento.

| Experimento | O que o agente decide | Oponente inicial | Métrica principal |
| --- | --- | --- | --- |
| Ataque | Próximo tiro. | Frota `random_legal` oculta. | Tiros válidos até vitória. |
| Posicionamento | Onde colocar cada navio. | Suíte fixa de atacantes. | Tiros válidos até afundamento. |

Os dois experimentos serão executados nos cenários `battleship` e
`periodic-table-battleship`.

## Experimento 1: aprender a atacar

### Objetivo

Treinar uma política para escolher o melhor próximo tiro considerando somente a
grade, os acertos e os erros anteriores. A frota inimiga muda a cada episódio,
mas é sempre criada pela política `random_legal` definida na especificação.

Em um tabuleiro sem tiros, a política pode aprender um prior espacial. Após um
acerto, deve aprender a explorar os vizinhos ortogonais e inferir orientações
prováveis. É justamente essa combinação de busca global e exploração local que
queremos comparar entre os dois cenários.

### Técnicas

| Técnica | Papel | Decisão |
| --- | --- | --- |
| Aleatório mascarado | Piso de desempenho. | Obrigatório. |
| Hunt-target | Heurística explicável. | Obrigatório. |
| Q-learning tabular | Checagem didática de update off-policy. | Somente grades reduzidas, como 4 × 4. |
| SARSA tabular | Checagem didática de update on-policy. | Somente grades reduzidas, como 4 × 4. |
| MaskablePPO | Primeiro método principal. | Cenários completos. |
| DQN mascarado | Comparação off-policy futura. | Após o MaskablePPO. |

Q-learning e SARSA não são adequados como solução principal. O estado contém o
histórico de acertos e erros de até 118 células, o que torna uma tabela de
valores inviável e não permite generalização espacial. Eles ainda são úteis em
uma versão pequena do jogo: validam recompensas, máscara e convergência antes
da rede neural.

MaskablePPO é a escolha inicial mais segura para os tabuleiros reais. Ele
suporta ação discreta e máscara de ações inválidas, exatamente o que o problema
exige. Começaremos com uma política MLP que achata os três canais da observação.
Uma CNN será uma ablação posterior para verificar se explorar a vizinhança
espacial melhora o resultado.

DQN é interessante por ser off-policy e poder reutilizar experiências, mas a
implementação usada terá de mascarar ações tanto na escolha epsilon-greedy
quanto no máximo do alvo de Bellman. Não usaremos um DQN sem essa garantia.

## Experimento 2: aprender a posicionar

### Objetivo

Treinar uma política defensiva para escolher posições que façam uma frota
demorar mais para ser encontrada.

O episódio terá cinco decisões, uma por navio, em ordem de tamanho `5, 4, 3,
3, 2`. Cada ação representa uma colocação candidata de um navio, formada por
âncora e orientação. A máscara remove candidatos fora da topologia ou que se
sobreponham a navios já colocados. Depois do quinto navio, o ambiente simula o
ataque contra a frota completa.

O espaço será `Discrete(360)`: os índices `0..179` representam âncora com
orientação horizontal e `180..359`, a mesma âncora com orientação vertical. A
observação será uma `Box` de três canais: topologia, células já ocupadas e o
tamanho normalizado do próximo navio. O estado não inclui a identidade do
atacante sorteado, para que a política aprenda uma defesa robusta contra a
mistura, e não uma resposta a um rótulo de oponente.

### Oponente e recompensa

O primeiro treino usará uma mistura fixa de atacantes:

- aleatório mascarado;
- hunt-target;
- atacante MaskablePPO congelado no Marco 3.

Cada episódio sorteia um desses atacantes com pesos registrados na configuração.
O posicionador recebe como recompensa o número normalizado de tiros válidos que
a frota sobreviveu. A avaliação informa o resultado contra cada atacante
separadamente e contra a média da suíte. Dessa forma, uma frota não parece boa
apenas porque explora uma fraqueza do atacante aleatório.

Antes do atacante PPO estar disponível, a mistura contém apenas os dois
baselines. Cada versão congelada de atacante, seus pesos, desempates, seeds e
hash do checkpoint entram no manifesto da execução. As métricas defensivas
incluem tiros até o primeiro acerto, primeiro navio afundado, afundamento total
e curva de segmentos restantes.

MaskablePPO também será o método principal nesse experimento. O espaço é
discreto e tem muitas colocações ilegais em cada etapa. Q-learning e SARSA
ficam restritos a um tabuleiro miniatura pela mesma razão do experimento de
ataque.

## Visualizações e artefatos

Os dados brutos serão salvos por episódio em CSV ou Parquet. Pandas fará a
agregação e Seaborn, apoiado em Matplotlib, produzirá gráficos estáticos. Cada
figura terá cenário, experimento, versão do ambiente, agente, seed e intervalo
de confiança identificados no arquivo.

| Artefato | Técnica | Pergunta respondida |
| --- | --- | --- |
| Tabela Markdown e CSV | Pandas. | Qual agente vence com menos tiros? |
| Curva de aprendizado | `seaborn.lineplot` com intervalo de confiança por seed. | O agente convergiu e com que variância? |
| Boxplot ou violinplot | Seaborn. | Como se distribuem tiros até vitória ou afundamento? |
| Heatmap de tiros | `seaborn.heatmap` sobre a forma do tabuleiro. | Onde o atacante concentra busca e acertos? |
| Heatmap de segmentos | `seaborn.heatmap` sobre a forma do tabuleiro. | Onde o posicionador deixa sua frota? |
| Curva de sobrevivência | Seaborn. | Qual fração de frotas permanece viva após cada tiro? |
| GIF de episódio | Matplotlib com Pillow ou ImageIO. | Como a política toma decisões em uma partida concreta? |

Os GIFs mostrarão uma partida de avaliação com seed de demonstração, separada
das seeds de teste. Cada quadro exibirá acertos, erros, tiro atual, recompensa
acumulada e número do passo. A frota inimiga só será revelada no quadro final,
para não induzir a impressão de que o agente vê informação secreta.

Também serão gerados GIFs de progresso: o mesmo episódio de demonstração será
reexecutado com checkpoints do início, meio e fim do treinamento. Isso mostra
mudança de comportamento sem usar o conjunto de teste como material de ajuste.

O esquema de diretórios, escalas de cor, tratamento de lacunas e metadados de
cada artefato está em
[Contratos e critérios de aceite](07-contratos-e-criterios-de-aceite.md).

## Self-play posterior

Quando os dois experimentos estiverem validados, poderemos alternar o treino de
atacante e posicionador. Como isso transforma o problema em jogo multiagente por
turnos, a interface recomendada passa a ser PettingZoo AEC, mantendo adaptadores
Gymnasium para avaliações de agente único. A política será avaliada contra um
conjunto de versões históricas do oponente, não apenas contra a versão mais
recente, para reduzir ciclos de exploração entre as duas políticas.
