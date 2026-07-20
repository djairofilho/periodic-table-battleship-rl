# Ablação v0.4 de ataque

Menos `valid_shots` é melhor. A unidade estatística é o seed cego, após
a média das políticas treinadas no mesmo seed. As intervenções não revelam
a frota: usam somente observação pública e máscara de ações.

| Braço | Hipótese |
| --- | --- |
| control-v03 | Controle: reproduz recompensa e observação da v0.3. |
| exploration-reward | Reduzir a penalidade de erro de -1,0 para -0,2 permite ao PPO explorar e converter informação local de acertos em menos tiros. |
| available-channel | Expor o plano binário de ações ainda disponíveis melhora a estimativa de valor sem revelar a frota e reduz tiros válidos. |

## Comparações cegas contra o controle

| Candidata − controle | Diferença | IC 95% | Conclusão |
| --- | ---: | ---: | --- |
| available-channel − control-v03 | -0.14 | [-1.02; +0.77] | inconclusive |
| exploration-reward − control-v03 | +0.62 | [-0.28; +1.54] | inconclusive |

Um intervalo abaixo de zero favorece a candidata; um intervalo que cruza
zero é inconclusivo. A conclusão aplica-se ao orçamento e aos seeds deste
protocolo, não constitui ajuste posterior nem prova de generalização.
