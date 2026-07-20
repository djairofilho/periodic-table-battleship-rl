# Decisão de escala: CPU, GPU e a próxima campanha

Esta decisão responde à issue [#44](https://github.com/djairofilho/periodic-table-battleship-rl/issues/44) sem transformar um resultado do teste cego em motivo para aumentar orçamento.

## Evidência disponível em 20 de julho de 2026

| Item | Medição |
| --- | --- |
| Computador de referência | Acer Nitro AN515-54, i5-9300H, 32 GB RAM, GTX 1650 |
| PyTorch instalado | `2.13.0+cpu` |
| CUDA detectado pelo PyTorch | Não (`torch.cuda.is_available() == false`) |
| Dispositivos CUDA utilizáveis | 0 |
| Throughput CPU, ataque | 472 passos/s (uma thread, v0.3) |
| Throughput CPU, posicionamento | 360 passos/s (uma thread, v0.3) |

O hardware físico inclui uma GPU, mas a instalação efetivamente usada não tem
CUDA. Portanto não existe uma comparação CPU/GPU honesta nesta máquina: marcar
`device="cuda"` produziria erro ou cairia silenciosamente em CPU. A campanha
v0.3 e as ablações continuam explicitando o dispositivo em seus metadados.

## Decisão para esta release

Não iniciar uma campanha ampliada em GPU. A análise v0.3 mostrou que o PPO é
pior que `hunt_target-v1` em todas as topologias e não demonstrou vantagem
robusta no posicionamento. Escalar a mesma configuração não é uma hipótese de
melhoria, seria apenas custo adicional. A prioridade técnica correta é medir
as ablações pré-registradas da issue #43 e manter o teste cego isolado.

## Gate para uma campanha GPU futura

A campanha só pode ser proposta em uma issue nova quando houver uma variante
escolhida exclusivamente por treino e validação. A issue deve congelar:

1. a hipótese e a alteração mínima que venceu na validação;
2. o mesmo conjunto de seeds de treino/validação da v0.3 e um teste final novo,
   ainda não lido;
3. orçamento de passos, checkpoints e teto de horas;
4. versões de Python, PyTorch, CUDA, driver, `uv.lock` e commit;
5. a comparação de 10 mil passos CPU versus GPU com `device="cpu"` e
   `device="cuda"`, mesma seed, `n_steps`, `batch_size` e uma thread CPU;
6. throughput, pico de memória GPU e verificação de que as métricas de
   validação permanecem comparáveis dentro da variação entre seeds.

Para tornar a comparação válida, a GPU precisa estar disponível para o mesmo
ambiente `uv`; `torch.cuda.is_available()` deve retornar `true` antes de iniciar
qualquer treino. Caso uma mudança de instalação seja necessária, ela deve ser
registrada como mudança de ambiente, não misturada aos resultados da política.

## Próxima decisão

Depois das ablações e da matriz cross-topology, comparar apenas as médias e
intervalos de validação pré-especificados. Se nenhuma variante superar
`hunt_target-v1` de forma robusta, encerrar a linha PPO atual e investigar uma
representação ou algoritmo diferente, em vez de aumentar a escala.
