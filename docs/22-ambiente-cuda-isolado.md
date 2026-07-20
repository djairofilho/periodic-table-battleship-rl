# Ambiente CUDA isolado

Esta página executa a issue [#66](https://github.com/djairofilho/periodic-table-battleship-rl/issues/66): disponibilizar um ambiente CUDA reproduzível para a GTX 1650 sem alterar a venv CPU que define os resultados publicados.

## Estado verificado em 20 de julho de 2026

| Item | Evidência |
| --- | --- |
| GPU física | NVIDIA GeForce GTX 1650, 4.096 MiB de VRAM |
| Driver NVIDIA | 591.74; `nvidia-smi` anuncia compatibilidade com CUDA 13.1 |
| Ambiente publicado | `.venv`, PyTorch `2.13.0+cpu`, CUDA indisponível |
| Ambiente isolado validado | `.venv-cuda`, PyTorch `2.13.0+cu130`, CUDA disponível |
| Wheel CUDA escolhido | `torch==2.13.0+cu130`, CPython 3.11, Windows x86-64 |
| Origem do wheel | índice oficial `https://download.pytorch.org/whl/cu130` |
| Isolamento | `.venv-cuda` é separada, não altera `pyproject.toml`, `uv.lock` nem `.venv` |

O driver é compatível com o runtime CUDA 13.0 do wheel. A GPU tem capacidade de computação 7.5, apropriada para redes pequenas e batches moderados, mas os 4 GB de VRAM exigem medir memória antes de paralelizar treinos.

## Instalação reproduzível

No PowerShell, a partir da raiz do repositório:

```powershell
uv venv .venv-cuda --python 3.11
$cudaPython = ".venv-cuda\Scripts\python.exe"
uv pip install --python $cudaPython --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0+cu130"
uv pip install --python $cudaPython -e ".[train,visual]" pytest ruff
```

O primeiro download tem aproximadamente 1,78 GiB. Ele deve terminar antes de prosseguir; não substitua `torch` na `.venv` do projeto e não gere um novo `uv.lock` para esse ambiente experimental.

Se a transferência for interrompida, recrie somente a venv isolada antes de
tentar novamente:

```powershell
uv venv .venv-cuda --clear --python 3.11
```

## Verificação e evidência

Execute o verificador com a venv CUDA, nunca via `uv run` sem indicar a venv:

```powershell
$cudaPython = ".venv-cuda\Scripts\python.exe"
& $cudaPython scripts/verify_cuda_environment.py `
  --require-cuda `
  --out artifacts/v0.6-cuda-readiness/cuda-readiness.json
```

O comando falha se `torch.cuda.is_available()` for falso. Quando bem-sucedido, o JSON preserva a versão de Python e PyTorch, runtime CUDA, inventário do driver, nome e capacidade da GPU, além de uma operação real de tensor na GPU.

## Estado desta execução

Em 20 de julho de 2026, o wheel foi instalado exclusivamente em `.venv-cuda`
e `scripts/verify_cuda_environment.py --require-cuda` passou. A evidência
versionada confirma uma operação real na GTX 1650, PyTorch `2.13.0+cu130` e
runtime CUDA 13.0 em
[`artifacts/v0.6-cuda-readiness/cuda-readiness.json`](../artifacts/v0.6-cuda-readiness/cuda-readiness.json).
O ambiente publicado `.venv` continua CPU-only de propósito; ele não foi
alterado nem usado para produzir a medição CUDA.

## Prontidão para a issue #67

O benchmark CPU/GPU só pode começar depois de existir um `cuda-readiness.json` com `"cuda_ready": true`. A execução deve usar o mesmo commit, seed, número de passos, batch e `n_steps` em ambos os dispositivos, registrar tempo, passos por segundo e pico de memória, e manter a campanha de resultados publicada na `.venv` CPU intacta.
