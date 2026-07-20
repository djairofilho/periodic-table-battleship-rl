# Protocolo de benchmark

## Escopo

Mantidas a frota, a regra de posicionamento e o algoritmo, como se comporta a
busca em `periodic-table-battleship` quando comparada a `battleship`?

A comparação principal muda simultaneamente a topologia e o número de células
válidas, de 100 para 118. Portanto, ela mede o efeito combinado dessas duas
propriedades. Uma ablação `dense-118`, com 118 células sem lacunas internas,
será exigida antes de atribuir uma diferença exclusivamente à geometria.

Este protocolo cobre o experimento de ataque. O posicionamento tem um protocolo
próprio em [Experimentos e visualizações](05-experimentos-e-visualizacoes.md).

## Agentes da primeira rodada

| Agente | Papel | Regra |
| --- | --- | --- |
| `random` | Piso de referência | Escolhe uniformemente entre ações mascaradas. |
| `hunt_target` | Baseline heurístico | Após um acerto não resolvido, prioriza vizinhos ortogonais ainda válidos; caso contrário, escolhe aleatoriamente. |
| `maskable_ppo` | Primeiro baseline de RL | Usa `MaskablePPO` e a máscara provida pelo ambiente. |

O baseline `hunt_target` é deliberadamente simples. Ele testa se o agente de
RL aprende a explorar a contiguidade local sem receber essa regra codificada.

## Reprodutibilidade

- Treinar cada combinação de cenário e agente em 10 sementes.
- Fixar uma lista pública de sementes de treino, validação e teste.
- Separar as sementes de teste antes de qualquer ajuste de hiperparâmetros.
- Congelar versão do ambiente, configuração, dependências e `uv.lock` em cada
  execução reportada.
- Salvar configuração, sementes, métricas por episódio e resumo agregado em
  diretórios versionados de resultados.

O mesmo orçamento de passos de ambiente e a mesma família de hiperparâmetros
devem ser usados nos dois cenários para cada agente. Ajustes exclusivos de um
cenário devem ser identificados como ablação, não como resultado principal.

## Métricas

| Métrica | Definição | Direção |
| --- | --- | --- |
| Tiros válidos até vitória | Número de células novas chamadas até os 17 acertos. | Menor é melhor. |
| Taxa de vitória | Episódios terminados por vitória antes do limite. | Maior é melhor. |
| Acerto por tiro | Acertos divididos por tiros válidos. | Maior é melhor. |
| AUC de descoberta | Área sob a curva de segmentos encontrados por tiro. | Maior é melhor. |
| Ações inválidas | Média de ações fora da máscara. | Zero é o esperado. |

Os resultados devem trazer média, desvio-padrão, mediana e intervalo de
confiança por semente. Não reportar somente o melhor checkpoint.

As métricas são calculadas por seed, não tratando episódios de uma mesma seed
como amostras independentes. Além de tiros brutos, serão reportados tiros
normalizados por `N` e excesso sobre os 17 segmentos da frota. Fórmulas,
tratamento de truncamento e estatística estão definidos em
[Contratos e critérios de aceite](07-contratos-e-criterios-de-aceite.md).

## Avaliação

1. Selecionar checkpoint apenas com a partição de validação.
2. Avaliar o checkpoint fixado em todas as sementes de teste.
3. Usar a mesma coleção de frotas por semente para todos os agentes.
4. Publicar configuração e resultados brutos suficientes para reproduzir as
   tabelas e gráficos.

## O que não comparar ainda

Não comparar desempenho contra a CPU da interface, pois ela segue regras de
turno distintas. Também não comparar com políticas que veem a posição dos
navios, pois isso viola a observação parcial central do problema.
