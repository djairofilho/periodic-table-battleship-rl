# Roadmap

## Marco 0: fundação e protocolo

- [x] Criar projeto Python gerenciado por uv.
- [x] Definir licença MIT, CI e testes mínimos.
- [x] Registrar regras de equivalência e protocolo de benchmark.
- [x] Pesquisar Gymnasium, action masking, uv e dados químicos.
- [ ] Congelar contratos de topologia, observação, métricas e artefatos.
- [ ] Criar Milestones, épicos e issues atômicas no GitHub.

## Marco 1: ambiente verificável

- [ ] Implementar a topologia clássica e a topologia periódica.
- [ ] Implementar o controle `dense-118` para separar cardinalidade de lacunas.
- [ ] Versionar o catálogo de elementos com fonte IUPAC.
- [ ] Implementar amostrador de frotas legais e testes de propriedades.
- [ ] Implementar `reset`, `step`, renderização textual e registro Gymnasium.
- [ ] Executar `check_env` e testes determinísticos por semente.

## Marco 2: baselines

- [ ] Implementar agentes `random` e `hunt_target`.
- [ ] Definir arquivos de configuração e separação de sementes.
- [ ] Gerar o primeiro relatório dos dois baselines.

## Marco 3: experimento de ataque

- [ ] Implementar Q-learning e SARSA em tabuleiros reduzidos para validar a dinâmica e a máscara.
- [ ] Adicionar extra `train` e pipeline para MaskablePPO.
- [ ] Treinar o atacante em ambos os cenários com posicionamentos `random_legal`.
- [ ] Avaliar em seeds nunca usadas no ajuste.
- [ ] Publicar tabelas, curvas, heatmaps e GIFs das partidas avaliadas.

## Marco 4: experimento de posicionamento

- [ ] Implementar ambiente sequencial de posicionamento com máscara de colocações legais.
- [ ] Avaliar contra uma suíte fixa de atacantes aleatório e hunt-target.
- [ ] Treinar MaskablePPO para maximizar tiros de sobrevivência.
- [ ] Comparar distribuições espaciais de frotas e curvas de sobrevivência.

## Marco 5: extensões

- [ ] Atributos químicos observáveis como novo cenário.
- [ ] Regra opcional de não-adjacência.
- [ ] Self-play entre atacante e posicionador.
- [ ] Integração opcional com a interface do jogo de origem.
