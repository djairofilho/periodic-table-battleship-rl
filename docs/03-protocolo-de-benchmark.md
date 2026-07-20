# Protocolo de benchmark

## Pergunta

Mantidas a frota, a regra de posicionamento e o algoritmo, qual é o efeito da
topologia da tabela periódica sobre a eficiência de busca quando comparada ao
tabuleiro clássico?

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
