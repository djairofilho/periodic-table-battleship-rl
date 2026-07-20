# Benchmark CPU/GPU e campanha escalável

Esta página executa a issue [#67](https://github.com/djairofilho/periodic-table-battleship-rl/issues/67) e prepara a [#68](https://github.com/djairofilho/periodic-table-battleship-rl/issues/68). Ela não altera a venv CPU, os resultados publicados ou o teste cego.

## O que é medido

O script `scripts/benchmark_cpu_gpu.py` mede pares CPU/CUDA para as três arquiteturas candidatas:

| Arquitetura | Implementação medida | Operação |
| --- | --- | --- |
| CNN | Backbone espacial da PPO-CNN, com cabeça de ações | forward, backward e Adam |
| DQN | MLP Q-network da DQN mascarada | forward, backward e Adam |
| GNN | Message passing sobre o grafo da topologia periódica | forward, backward e Adam |

O lote, seed, quantidade de aquecimento, atualizações medidas e número de threads CPU são idênticos nos dois dispositivos. Pesos e dados nascem na CPU, com a mesma seed, antes da cópia para CUDA. O JSON registra commit, hash de `uv.lock`, Python, PyTorch, runtime CUDA, GPU, VRAM total, throughput e picos de memória CUDA.

É um **microbenchmark neural**, não uma medida de episódios por segundo nem uma alegação de melhora de política. Ele exclui stepping do ambiente, coleta de rollout, avaliação e escrita em disco. Esses itens serão medidos na campanha apenas depois de uma candidata superar o gate de validação.

## Execução reproduzível

Primeiro confirme a prontidão do ambiente isolado descrito em [Ambiente CUDA isolado](22-ambiente-cuda-isolado.md):

```powershell
$cudaPython = ".venv-cuda\Scripts\python.exe"
& $cudaPython scripts/verify_cuda_environment.py --require-cuda `
  --out artifacts/v0.6-cuda-readiness/cuda-readiness.json
```

Depois execute o benchmark, ainda com a mesma venv:

```powershell
& $cudaPython scripts/benchmark_cpu_gpu.py `
  --scenario periodic-table-battleship `
  --seed 8601 `
  --batch-size 32 `
  --warmup-iterations 5 `
  --measured-iterations 20 `
  --cpu-threads 1 `
  --out artifacts/v0.6-cuda-benchmark/cpu-gpu-microbenchmark.json
```

O comando sai com código `2` e uma mensagem explícita se CUDA não estiver pronta. Ele nunca aceita `device="cuda"` e executa secretamente em CPU, nem publica um relatório CPU/GPU parcial.

## Gate para a campanha escalável (#68)

Uma campanha ampliada só pode ser aberta quando todos os itens abaixo forem verdadeiros:

1. `cuda-readiness.json` tem `cuda_ready: true` e registra uma operação real na GPU.
2. O microbenchmark pareado foi gerado no mesmo commit e possui resultados para CNN, DQN e GNN em CPU e CUDA.
3. Uma candidata foi promovida exclusivamente pela validação multi-seed, sem abrir o teste cego.
4. A issue da campanha congela arquitetura, topo, seed, orçamento, checkpoints, limite de horas, VRAM máxima permitida e a regra de interrupção por OOM.
5. A campanha registra tanto tempo de treino ponta a ponta quanto o resultado pareado contra `hunt-target`; velocidade não é critério de promoção.

## Resultado registrado

A venv isolada `.venv-cuda` foi validada em uma NVIDIA GeForce GTX 1650 de
4 GiB, com PyTorch `2.13.0+cu130`, runtime CUDA 13.0 e capacidade 7.5. A
prontidão e o microbenchmark estão em
[`artifacts/v0.6-cuda-readiness`](../artifacts/v0.6-cuda-readiness) e
[`artifacts/v0.6-cuda-benchmark`](../artifacts/v0.6-cuda-benchmark).

| Arquitetura | Atualizações/s CPU | Atualizações/s CUDA | Razão CUDA/CPU |
| --- | ---: | ---: | ---: |
| CNN | 41,80 | 248,31 | 5,94× |
| DQN MLP | 361,39 | 314,81 | 0,87× |
| GNN | 51,40 | 163,46 | 3,18× |

O lote e a carga foram idênticos nos dois dispositivos. A GPU ajuda nas
arquiteturas CNN e GNN desta carga, mas a DQN MLP pequena continua mais rápida
na CPU, pois o custo de lançamento/transferência domina. É um microbenchmark
neural, não uma alegação de ganho de qualidade de agente.

Os itens 1 e 2 do gate foram satisfeitos. O item 3 não foi: a ablação da
CNN com crença pública teve **97,22** tiros contra **96,89** do controle na
validação, onde menos é melhor. Logo a candidata não foi promovida, nenhum
teste cego foi aberto e a campanha ampliada da issue #68 é formalmente
**rejeitada nesta release**, não apenas adiada. Isso preserva o teste final e
evita gastar GPU numa hipótese sem sinal de ganho.
