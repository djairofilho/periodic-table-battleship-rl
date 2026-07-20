# Contratos e critérios de aceite

## Escopo e versão

`topology-v1` define os cenários `battleship` e
`periodic-table-battleship`. Alterar a máscara de células, a adjacência, a
codificação de ação ou a observação exige uma nova versão de ambiente.

## Topologia

Cada cenário usa um canvas de 10 linhas por 18 colunas. A ação de tiro `a` é o
índice zero-based `linha * 18 + coluna`. O artefato de topologia versionado
deverá conter a matriz de 180 posições e, no cenário periódico, a associação
entre ação, elemento, período, grupo e série.

- `battleship`: as 100 células do retângulo 10 × 10 são válidas.
- `periodic-table-battleship`: exatamente 118 células são válidas.
- Lacunas nunca são células, alvo ou trecho de navio.
- Vizinhos são somente arestas ortogonais entre duas células válidas.
- As séries f são componentes desconectados das linhas principais. Navios não
  podem saltar entre elas.
- Navios são lineares, horizontais ou verticais, sem sobreposição e com contato
  permitido.

`dense-118` será um cenário de controle posterior com 118 células válidas e
sem lacunas internas. Ele não altera o benchmark principal, mas é obrigatório
para atribuir uma diferença exclusivamente à topologia em vez da cardinalidade.

## Frota e amostragem

A frota é `[5, 4, 3, 3, 2]`, com 17 segmentos. Os dois navios de tamanho 3 são
canonizados como `ship-3a` e `ship-3b`, evitando estados equivalentes ambíguos.

`random_legal-v1` posiciona navios nessa ordem e amostra uniformemente entre as
colocações candidatas legais em cada etapa. A política não é uniforme sobre o
conjunto total de frotas completas. Seu algoritmo, seed, versão e a lista de
frotas de teste entram em um manifesto imutável. Se uma tentativa não permitir
completar a frota, o amostrador recomeça com a mesma fonte aleatória até obter
uma configuração legal; a quantidade de reinícios também é registrada.

## Ambiente de ataque

### Contrato público

- Ação: `Discrete(180)` e `action_masks()` com ações ainda não chamadas.
- Observação: quatro canais definidos na especificação do ambiente.
- `info` público: `is_hit`, `sunk_ship_length`, `valid_shots`,
  `invalid_attempts` e identificador de episódio. Nunca inclui ocupação oculta.
- Recompensa: perfil `efficiency-v0` já documentado.
- Vitória: todos os 17 segmentos atingidos.
- Uma política que respeita a máscara termina por vitória em no máximo `N`
  tiros válidos. Para robustez, `max_total_attempts = 2 * N`; após esse limite,
  a partida é truncada.

### Critérios de aceite

- `check_env` aprovado.
- Mesmo `reset(seed)` produz a mesma frota e mesma observação inicial.
- A máscara contém todas e somente as ações legais.
- Nenhuma ação ou observação revela segmentos não atingidos.
- Em amostra automatizada, 100% das frotas são legais e partidas mascaradas
  terminam em no máximo `N` tiros válidos.

## Ambiente de posicionamento

- Ordem das decisões: `5, 4, 3a, 3b, 2`.
- Ação: `Discrete(360)`. `0..179` usa orientação horizontal; `180..359`,
  vertical. A âncora é `ação % 180`.
- Observação: canais de topologia, ocupação atual e tamanho normalizado do
  próximo navio.
- Máscara: somente colocações completas, dentro da topologia e sem
  sobreposição.
- Recompensa intermediária: `0` para uma colocação válida. Recompensa terminal:
  `tiros_válidos_até_afundamento / N`. Ação inválida: `-1`.
- O atacante é sorteado da mistura congelada no `reset(seed)`. O manifesto
  guarda pesos, versões, checkpoint, desempate e seed de cada atacante.

### Critérios de aceite

- Toda frota construída pela máscara é legal.
- A mesma seed seleciona o mesmo atacante, frota e resultado sob uma política
  determinística.
- O baseline `random_legal` é avaliado contra cada atacante e contra a mistura.
- A avaliação reporta primeiro acerto, primeiro afundamento, afundamento total
  e segmentos restantes por tiro.

## Métricas e estatística

Para `N` células válidas, `h_t` é o número cumulativo de segmentos atingidos
após `t` tiros válidos. A área de descoberta é:

```text
AUC_descoberta = sum(h_t para t de 1 a N) / (17 * N)
```

Após término, `h_t` mantém seu último valor até `N`. Em episódios truncados,
isso penaliza explicitamente a descoberta incompleta. Também serão publicados:

- tiros válidos até vitória ou afundamento;
- tiros normalizados por `N` e excesso sobre 17;
- taxa de acerto, ações inválidas, vitória e truncamento;
- curva de sobrevivência, definida como a fração de frotas com ao menos um
  navio não afundado após cada tiro.

Treino, validação e teste usam listas disjuntas de seeds. Cada avaliação guarda
configuração, commit, hash de `uv.lock`, Python, SO, dependências, hardware,
manifesto de episódios e checkpoint. A estatística é agregada primeiro por
seed. O relatório inclui diferença pareada contra baseline, intervalo de
confiança bootstrap e número de episódios por seed.

## Dados, gráficos e GIFs

Na fase atual, cada execução reproduzível usa este formato:

```text
runs/<run-id>/
  manifest.json
  episodes.jsonl
  summary.json
```

`manifest.json` contém a configuração, proveniência de software e hardware,
hash de `uv.lock` e inventário ordenado dos episódios. `episodes.jsonl` é UTF-8
canônico, um resultado público por linha, gravado atomicamente antes do
manifesto. `summary.json` agrega primeiro por seed. Checkpoints, tabelas,
figuras e GIFs serão adicionados ao diretório da execução quando os pipelines
de treinamento e visualização estiverem disponíveis.

Heatmaps usam `NaN` nas lacunas e registram se medem tiros, acertos ou taxa de
acerto condicional. Comparações diretas usam a mesma escala de cor. Curvas
registram agregação, intervalo de confiança e qualquer suavização; os dados
sem suavização permanecem disponíveis. Todo GIF informa renderer, versão, seed,
política e checkpoint no manifesto e revela a frota adversária somente no
quadro final.

## Publicação de um resultado

Um resultado só pode ser publicado quando um comando de reprodução recriar suas
tabelas, figuras e GIFs a partir do manifesto e dos resultados brutos, sem
usar dados de teste para ajuste.
