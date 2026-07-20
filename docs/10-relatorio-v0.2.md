# Relatório v0.2: campanha piloto controlada

## Escopo executado

Foram treinados 15 checkpoints MaskablePPO, sempre um por vez: nove de ataque
(três seeds em cada uma das três topologias) e seis de posicionamento (três
seeds nos cenários clássico e periódico). Cada checkpoint recebeu 2.048 passos
de ambiente. A seleção usou cinco seeds de validação e o teste final usou 20
seeds cegos, conforme o [protocolo v0.2](09-protocolo-campanha-v0.2.md).

O experimento de posicionamento incluiu uma mistura equiponderada de atacante
aleatório, hunt-target e PPO de ataque congelado. Logo, ele cobre os dois tipos
de aprendizado pretendidos: onde atirar e onde posicionar a frota.

## Ataque

Menos tiros válidos é melhor. O MaskablePPO não superou hunt-target com este
orçamento curto; seus intervalos bootstrap não cruzam zero, portanto a perda
observada é inequívoca dentro desta campanha.

| Cenário | PPO | Hunt-target | PPO − hunt | IC bootstrap 95% |
| --- | ---: | ---: | ---: | --- |
| `battleship` | 95,60 | 62,65 | +32,95 | [+27,15; +38,95] |
| `dense-118` | 110,00 | 75,45 | +34,55 | [+26,40; +42,35] |
| `periodic-table-battleship` | 115,20 | 67,15 | +48,05 | [+40,40; +55,20] |

O controle `dense-118` também é pior que o tabuleiro clássico para ambas as
políticas, mas a tabela periódica é ainda mais difícil para o PPO neste
orçamento. Isso é uma observação descritiva, não uma atribuição causal final:
o próximo passo deve aumentar passos, seeds e explorar hiperparâmetros antes
de concluir que a irregularidade geométrica é a causa.

## Posicionamento

Mais tiros para afundar a frota é melhor. As políticas selecionadas concluíram
100% dos episódios de teste.

| Cenário | Hunt-target | PPO congelado | Mistura |
| --- | ---: | ---: | ---: |
| `battleship` | 67,30 | 97,00 | 86,60 |
| `periodic-table-battleship` | 64,20 | 116,00 | 97,60 |

Esses números descrevem a sobrevivência contra os componentes e a mistura que
definiram a recompensa; ainda falta uma baseline independente de
posicionamento e uma campanha maior para medir ganhos de RL de modo
comparativo.

## Artefatos e reexecução

- Dados e manifests: [`runs/v0.2-controlled`](../runs/v0.2-controlled)
- Tabelas e relatório de máquina: [`artifacts/v0.2-controlled`](../artifacts/v0.2-controlled)
- Gráfico de ataque: [`attack-test-comparison.png`](../artifacts/v0.2-controlled/figures/attack-test-comparison.png)
- GIF público de ataque: [`periodic-ppo-attack.gif`](../artifacts/v0.2-controlled/figures/periodic-ppo-attack.gif)
- Heatmap e GIF de posicionamento: [`figures`](../artifacts/v0.2-controlled/figures)

Para reexecutar, instale as dependências e rode:

```powershell
uv sync --all-groups --extra train --extra visual
uv run --extra train --extra visual python scripts/run_v0_2_campaign.py
```

Os checkpoints são locais por padrão, em `.local-runs/v0.2-controlled`; os
resultados e visualizações públicos são determinísticos dado o ambiente e os
artefatos de treinamento registrados nos manifests.
