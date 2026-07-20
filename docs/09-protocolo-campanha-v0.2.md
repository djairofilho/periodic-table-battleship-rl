# Protocolo da campanha v0.2

Esta campanha transforma os testes iniciais de fumaça em uma comparação
controlada. Ela mantém o estado privado das frotas fora dos artefatos públicos
e registra seeds, checkpoints, versões e resultados por episódio.

## Escopo

Ataque avalia MaskablePPO contra as baselines `random_masked-v1` e
`hunt_target-v1` em três geometrias:

| Cenário | Células válidas | Papel causal |
| --- | ---: | --- |
| `battleship` | 100 | referência retangular tradicional |
| `periodic-table-battleship` | 118 | geometria irregular da tabela periódica |
| `dense-118` | 118 | controle conectado, sem lacunas internas |

O experimento de posicionamento usa `battleship` e
`periodic-table-battleship`. O agente aprende contra uma mistura congelada e
equiponderada de atacante aleatório, hunt-target e um atacante PPO congelado
selecionado na validação de ataque do mesmo cenário.

## Orçamento e seeds

| Item | Valor |
| --- | --- |
| Seeds de treinamento | `1101`, `1102`, `1103` |
| Seeds de validação | `2101` a `2105` |
| Seeds de teste cegos | `3101` a `3120` |
| PPO por seed | 512 passos de ambiente |
| `n_steps` / `batch_size` | 256 / 64 |
| Episódios por seed de avaliação | 1 |
| Bootstrap | 10.000 reamostragens, IC percentil de 95% |

O orçamento é deliberadamente identificado como uma campanha piloto
controlada: é suficiente para revelar diferenças grosseiras e exercitar toda a
cadeia reproduzível, mas não substitui uma campanha de maior escala.

## Seleção e teste cego

1. Treinar três checkpoints independentes por cenário.
2. Avaliar cada checkpoint apenas nos cinco seeds de validação.
3. Selecionar o checkpoint com melhor média: menos tiros no ataque e mais
   tiros para afundar no posicionamento.
4. Congelar a seleção e avaliar o checkpoint escolhido nos 20 seeds de teste.
5. Rodar as baselines nos mesmos seeds de teste e comparar por pares de seed.

O script `scripts/run_v0_2_campaign.py` materializa a campanha. Checkpoints
locais ficam em `.local-runs/` e os registros públicos em `runs/` e
`artifacts/`.

## Métricas e interpretação

No ataque, menor `valid_shots` é melhor. No posicionamento, maior
`valid_shots_to_sink` é melhor. A comparação principal de ataque é PPO versus
`hunt_target-v1`, com diferença pareada por seed e intervalo bootstrap. O
controle `dense-118` permite separar o efeito de ter 118 células do efeito de
lacunas e desconexões na tabela periódica.

Não se deve interpretar intervalos desta campanha piloto como evidência final
de superioridade geral: eles refletem somente os seeds, orçamento e versões
acima.
