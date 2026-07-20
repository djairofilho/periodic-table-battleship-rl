# Protocolo v0.7: Bayes calibrado e destilação pública

## Objetivo

A v0.7 verifica se um planejador Bayesiano baseado somente no histórico
público é uma referência confiável nos três cenários e se uma política neural
mais barata pode imitá-lo sem perder a vantagem sobre `hunt-target`.

Este documento separa desenvolvimento, validação, teste cego e self-play.
Nenhum resultado de um estágio pode ser usado retroativamente para escolher
configurações de um estágio anterior.

## Candidatas e controles

| Papel | Identidade | Uso |
| --- | --- | --- |
| Controle forte | `hunt-target-v1` | Todos os cenários e splits |
| Professor | `belief_probability_mc-v1` | Amostragem pública calibrada |
| Estudante CNN | a definir pela issue #76 | Destilação com máscara |
| Estudante GNN | a definir pela issue #76 | Destilação com máscara |

O professor não acessa a frota privada. Ele recebe apenas observação pública,
máscara de ações e especificação conhecida da frota.

## Estágios e dependências

```text
#72 calibração MC ─┐
                  ├─> #73 validação Bayesiana ─┐
#74 contrato       │                             ├─> #77 gate/teste
                  └─> #75 dataset ─> #76 distilação ┘
                                                      └─> #78 self-play
                                                           └─> #79 release
```

## Split e seeds

As seeds de desenvolvimento podem ser usadas para smoke tests e para a
calibração exata do microtabuleiro. A seleção de configuração usa somente as
seeds de validação registradas pelo runner v0.7. O inventário de teste é novo,
gerado uma vez após congelar a candidata, e não pode aparecer em código de
treino, logs de busca de hiperparâmetros ou artefatos de validação.

## Gate de promoção

Uma única estudante pode ser promovida ao teste cego somente se todos forem
verdadeiros:

1. a calibração Monte Carlo reporta seu erro contra o posterior exato do
   microtabuleiro e não omite falhas de amostragem;
2. a política completa vence `hunt-target-v1` em tiros válidos médios em pelo
   menos dois dos três cenários de validação;
3. a diferença pareada estudante menos `hunt-target` tem intervalo bootstrap
   de 95% inteiramente abaixo de zero nos cenários em que se alega ganho;
4. taxa de vitória é 1,0, tentativas inválidas são zero e não há truncamento;
5. arquitetura, checkpoint, orçamento, dataset, seeds, dispositivo e regra de
   desempate foram congelados antes de criar as seeds cegas.

Se nenhuma candidata passar, a issue #77 será fechada como `not planned` para
esta release. Isso é resultado válido e não autoriza abrir o teste cego.

## Teste cego e self-play

O teste cego executa uma vez a candidata promovida e os controles nos três
cenários. Não há retreinamento, ajuste de amostras ou troca de checkpoint após
ver seus resultados. Self-play só começa depois do teste e somente contra o
atacante promovido, mantendo uma suíte congelada com Bayesiano e `hunt-target`.

## Medidas e publicação

O placar principal usa tiros válidos até vencer, onde menos é melhor. Serão
publicados também taxa de vitória, AUC de descoberta, primeiro acerto,
primeiro navio afundado, latência por decisão, memória e custo de treino.
Cada execução preserva commit Git, hash de `uv.lock`, configuração, seeds,
split e artefatos públicos. Gráficos não substituem as tabelas e JSONs.
