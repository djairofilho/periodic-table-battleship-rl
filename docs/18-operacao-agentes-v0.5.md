# Operação paralela v0.5 e perfis de agentes

Esta operação aproveita GPT-5.3-Codex-Spark para reduzir trabalho de apoio,
mantendo decisões de pesquisa, código de RL, execução e estatística sob um
agente principal responsável pela integração.

## Perfis e limites

| Perfil | Responsabilidades | Não pode decidir ou modificar | Limite |
| --- | --- | --- | --- |
| Agente principal | Contratos, código, treino, análise estatística, revisão, commits e publicação | Não delega decisão experimental a outro perfil. | Um por integração. |
| Especialista de algoritmo | Implementação isolada de CNN, DQN, imitação ou GNN e testes de unidade | README, `pyproject.toml`, contratos compartilhados, commits e publicação. | Um por diretório de algoritmo. |
| GPT-5.3-Codex-Spark | Inventário de artefatos, tabelas Markdown, links, revisão de consistência, triagem de logs e rascunhos de relatório | Código de RL, hiperparâmetros, resultados, decisões estatísticas, execução de treino, commits e publicação. | Até dois, somente em tarefas textuais independentes. |

O limite total é quatro agentes ativos, incluindo o agente principal. Treinos
pesados continuam serializados na máquina de referência; enquanto um treino
roda, os demais perfis podem preparar documentação, testes isolados ou revisar
artefatos já estáveis.

## Propriedade de arquivos

| Frente | Propriedade exclusiva | Saída para integração |
| --- | --- | --- |
| Protocolo | `src/.../protocol/`, `tests/protocol/`, `docs/17-*` | Contrato e testes. |
| PPO-CNN | Novo diretório próprio de arquitetura, seus testes e artefatos temporários | Patch, configuração e resumo técnico. |
| DQN | Novo diretório próprio de algoritmo, seus testes e artefatos temporários | Patch, configuração e resumo técnico. |
| Imitação | Novo diretório próprio de pré-treino, seus testes e artefatos temporários | Patch, configuração e resumo técnico. |
| GNN | Novo diretório próprio de grafo, seus testes e artefatos temporários | Patch, configuração e resumo técnico. |
| Spark | Somente `docs/drafts/` ou arquivo de handoff explicitamente designado | Tabela, lista de links ou triagem; nunca altera fonte compartilhada. |
| Integração | README, `pyproject.toml`, `uv.lock`, documentos finais e GitHub | Um único agente principal. |

Nenhum agente altera um arquivo fora de sua linha sem uma mensagem de handoff
explícita. Se duas frentes precisarem do mesmo contrato, o agente principal
faz a mudança antes de iniciar a onda seguinte.

## Exemplo de uma onda paralela

| Tarefa independente | Perfil | Entrada estável | Resumo obrigatório ao terminar |
| --- | --- | --- | --- |
| Implementar PPO-CNN | Especialista de algoritmo | Contrato v0.5 e interface de ataque | Arquivos, testes, configuração, limitação conhecida. |
| Inventariar visuais v0.4 | Spark | `artifacts/` e relatórios já publicados | Tabela de artefato, origem, legenda e link. |
| Rascunhar página de experimento | Spark | Protocolo e inventário, sem resultados novos | Estrutura Markdown e lacunas a preencher. |
| Integrar e validar | Agente principal | Os três resumos e patches | `ruff`, testes relevantes, revisão de proveniência e decisão. |

As três primeiras tarefas não compartilham arquivos graváveis. A integração é
sequencial e pertence exclusivamente ao agente principal.

## Checklist de handoff e integração

Antes do handoff, cada agente informa:

1. issue, hipótese e critério de aceite atendido;
2. arquivos criados ou modificados;
3. comandos executados e seus resultados;
4. artefatos produzidos, hashes ou localização;
5. limitações, decisões pendentes e arquivos que não tocou.

O agente principal então:

1. confirma que não houve sobreposição de propriedade;
2. revisa o diff e executa `git diff --check`;
3. executa `uv run ruff check .` e os testes aplicáveis;
4. verifica o contrato v0.5, seeds e proveniência antes de aceitar resultados;
5. organiza commits atômicos e só então atualiza issues, Project, README ou
   release.

Este fluxo permite usar o saldo do Spark em tarefas de alta velocidade e baixo
risco, sem transformar resumos textuais em decisões experimentais.
