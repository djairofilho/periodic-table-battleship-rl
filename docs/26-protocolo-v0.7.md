# Protocolo v0.9: controle, rastreabilidade e decisao de promoção

Este documento substitui e complementa as regras da v0.7 para fechar o ciclo v0.9.
Faz parte do fluxo de decisão para promoção de politica.

## Objetivo

Validar se uma politica melhora `hunt-target-v1` sob:

- split de treino/validacao sem vazamento;
- contrato de observacao publica;
- amostragem e métricas auditáveis;
- critério de gate para qualquer teste cego ou self-play.

## Candidatos no ciclo v0.9

| Papel | Id | Uso |
| --- | --- | --- |
| Controle | `hunt-target-v1` | Baseline em todos os cenarios |
| Professor | `belief_probability_mc-v1` | Planejamento Bayesiano com sampler calibrado |
| Estudantes | `bayesian-cnn-student-v1`, `bayesian-gnn-student-v1` | Destilação com demonstracoes do professor |

## Criterio de dados e sementes

- Cenario: `battleship`, `dense-118`, `periodic-table-battleship`.
- Treino e validacao com seeds registradas em cada artefato.
- Proibido usar dados da validacao para ajuste de hiperparametro.
- Todos os arquivos devem ter commit SHA, `git status`, parâmetros e manifest.

## Gate de promocao

Uma politica só pode ir para teste cego/teste aberto quando:

1. Melhorar em **pelo menos 2 cenarios**.
2. Diferença pareada `candidate - hunt-target` com intervalo de 95% todo abaixo de 0
   (meta de tiro valido, menor é melhor).
3. Sem truncamento e sem acordo inválido de ação em execução normal.
4. Criterios de custo e risco documentados em `docs/32-relatorio-v0.9.md`.

Se estes critérios não passarem, o ciclo é encerrado com decisão negativa documentada.

## Campanhas de fechamento v0.9

1. `v0.9-bayes-sampler-calibration`
2. `v0.9-bayes-sampler-ablation`
3. `v0.9-bayes-cross-topology-validation/smoke`
4. `v0.9-bayesian-students`
5. `v0.9-demonstrations`

## Evidência de referencia

- Relatorio de fechamento: `docs/32-relatorio-v0.9.md`
- Artefatos: `artifacts/v0.9-*`
- Manifesto de campanha: `artifacts/v0.9-demonstrations/dataset-manifest-v0.9.json`

## Entrega

Resultado final do ciclo:

- `hunt-target-v1` permanece baseline.
- `belief_probability_mc-v1` melhora em smoke, mas sem aprovação para promoção.
- Self-play permanece para ciclo seguinte (v0.10) após nova candidata aprovada no gate.
