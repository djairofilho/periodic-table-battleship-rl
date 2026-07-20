# Execução, rastreamento e trabalho paralelo

## Decisão

Usaremos documentação e GitHub juntos, com funções diferentes:

| Camada | Papel | Não deve conter |
| --- | --- | --- |
| Documentação versionada | Contratos, decisões de pesquisa, interfaces, métricas e protocolo. | Estado diário de tarefa. |
| Issue | Uma entrega pequena, verificável e atribuível. | Especificação técnica duplicada. |
| Milestone | Um corte de entrega que pode ser revisado ou publicado. | Dependências detalhadas. |
| GitHub Project | Fila e estado operacional. | Contratos de comportamento. |

Uma issue-pai com label `type: epic` organizará o trabalho, mas não substitui
as specs. Quando disponível, ela usará sub-issues e dependências nativas do
GitHub. O Project usa o fluxo padrão `Todo`, `In Progress` e `Done`; o label
`blocked` indica dependência pendente. Novas colunas só serão adicionadas quando
o volume de trabalho justificar a separação entre pronto e revisão.

## Estrutura no GitHub

### Milestones

1. `v0.1: ambientes e baselines reproduzíveis`
2. `v0.2: experimento de ataque`
3. `v0.3: experimento de posicionamento`
4. `v0.4: relatório visual e release`

### Épicos

- `Epic: benchmark reproduzível de Battleship e Periodic Table Battleship`
- `Epic: núcleo compartilhado e ambientes`
- `Epic: experimento de ataque`
- `Epic: experimento de posicionamento`
- `Epic: avaliação, visualizações e release`

### Labels

- Tipo: `type: epic`, `type: task`, `type: research`, `type: bug`.
- Área: `area: topology`, `area: env`, `area: baseline`, `area: rl`,
  `area: evaluation`, `area: visualization`, `area: docs`, `area: infra`.
- Experimento: `experiment: shared`, `experiment: attack`,
  `experiment: placement`.
- Priorização: `priority: high`, `priority: medium`, `priority: low`.
- Dependência: `blocked`.

O status fica somente no Project. Labels de status duplicariam informação e
costumam se desatualizar.

## Critério para uma issue pronta

Toda issue deve apontar à seção relevante da documentação e conter:

1. objetivo e fora de escopo;
2. arquivos ou diretórios sob sua propriedade;
3. dependências e issue-pai;
4. critérios de aceite verificáveis;
5. comando de validação;
6. artefato esperado, como teste, dataset, CSV, figura ou GIF.

O template em `.github/ISSUE_TEMPLATE/task.yml` padroniza esse conteúdo.

## Backlog inicial

| ID | Entrega | Depende de |
| --- | --- | --- |
| F1 | Congelar catálogo IUPAC, coordenadas e `topology-v1`. | Nenhuma |
| F2 | Implementar abstração de topologia dos dois cenários. | F1 |
| F3 | Implementar colocações legais e amostragem de frotas. | F2 |
| F4 | Implementar ambiente Gymnasium de ataque e máscara. | F2, F3 |
| F5 | Implementar renderização e serialização de episódios. | F4 |
| F6 | Testes de propriedades, seeds e `check_env`. | F3, F4 |
| B1 | Implementar `random_masked` e `hunt_target`. | F4 |
| B2 | Persistir configurações, seeds e resultados por episódio. | F4, B1 |
| B3 | Gerar benchmark inicial dos baselines. | B1, B2 |
| A1 | Criar microambiente para Q-learning e SARSA. | F2, F4 |
| A2 | Implementar e validar Q-learning e SARSA tabulares. | A1 |
| A3 | Criar pipeline MaskablePPO de ataque. | F4, B2 |
| A4 | Avaliar atacante em teste cego. | B1, A3 |
| A5 | Gerar tabelas, gráficos e GIFs de ataque. | F5, A4 |
| P1 | Implementar ambiente sequencial de posicionamento. | F2, F3 |
| P2 | Criar suíte fixa de atacantes defensivos. | P1, B1 |
| P3 | Integrar atacante PPO congelado à suíte. | P2, A3 |
| P4 | Criar pipeline MaskablePPO de posicionamento. | P1, P2 |
| P5 | Avaliar posicionador por atacante e mistura. | P3, P4 |
| P6 | Gerar gráficos e GIFs de posicionamento. | F5, P5 |
| R1 | Publicar relatório comparativo e release. | B3, A4, A5, P5, P6 |

## Estratégia de agentes paralelos

O limite operacional é um coordenador e até três agentes de execução. Como os
agentes compartilham o mesmo diretório, cada um recebe propriedade exclusiva de
diretórios e testes. O coordenador integra, altera `pyproject.toml`, registros
Gymnasium, README, contratos e commits.

```text
G0 contratos
  └─ G1 topologia + frota + schema de execução
       └─ G2 ambientes + baselines
            └─ G3 treinamento + relatórios
                 └─ G4 múltiplas seeds + release
```

| Onda | Agente A | Agente B | Agente C | Gate |
| --- | --- | --- | --- | --- |
| 0 | Coordenador: contratos. | Nenhum | Nenhum | G0 |
| 1 | `topology/` e testes. | `game/` e testes. | `evaluation/contracts.py`, configs e testes. | G1 |
| 2 | `envs/attack.py`. | `envs/placement.py`. | `policies/` e baselines. | G2 |
| 3 | Treino de ataque. | Treino de posicionamento. | `reporting/`, gráficos e GIFs. | G3 |
| 4 | Avaliação ataque. | Avaliação posicionamento. | Revisão estatística e relatório. | G4 |

Não treinaremos três políticas pesadas na mesma máquina ao mesmo tempo. Nesta
fase, um agente treina e os demais fazem avaliação, renderização ou revisão.

## Gates de integração

| Gate | Condição para avançar |
| --- | --- |
| G0 | Contratos registrados com exemplos de ação, máscara e observação. |
| G1 | `ruff`, `pytest`, 118 células válidas, índices bidirecionais e frotas legais por seed. |
| G2 | `check_env`, zero vazamento de segredo, baselines sem ação inválida. |
| G3 | Smoke training reproduzível, CSV válido e GIF sem revelar frota antes do fim. |
| G4 | Seeds disjuntas, resultados brutos preservados e gráficos reproduzíveis. |

Depois de cada gate, outro agente revisa o módulo que não implementou. Só o
coordenador integra e organiza commits atômicos.
