# Roadmap

## Marco 0: fundação e protocolo

- [x] Criar projeto Python gerenciado por uv.
- [x] Definir licença MIT, CI e testes mínimos.
- [x] Registrar regras de equivalência e protocolo de benchmark.
- [x] Pesquisar Gymnasium, action masking, uv e dados químicos.

## Marco 1: ambiente verificável

- [ ] Implementar a topologia clássica e a topologia periódica.
- [ ] Versionar o catálogo de elementos com fonte IUPAC.
- [ ] Implementar amostrador de frotas legais e testes de propriedades.
- [ ] Implementar `reset`, `step`, renderização textual e registro Gymnasium.
- [ ] Executar `check_env` e testes determinísticos por semente.

## Marco 2: baselines

- [ ] Implementar agentes `random` e `hunt_target`.
- [ ] Definir arquivos de configuração e separação de sementes.
- [ ] Gerar o primeiro relatório dos dois baselines.

## Marco 3: reinforcement learning

- [ ] Adicionar extra `train` e pipeline para MaskablePPO.
- [ ] Treinar em ambos os cenários com rastreio de métricas.
- [ ] Avaliar em seeds nunca usadas no ajuste.
- [ ] Publicar tabelas, curvas e limitações do experimento.

## Marco 4: extensões

- [ ] Atributos químicos observáveis como novo cenário.
- [ ] Regra opcional de não-adjacência.
- [ ] Agente de posicionamento e self-play.
- [ ] Integração opcional com a interface do jogo de origem.
