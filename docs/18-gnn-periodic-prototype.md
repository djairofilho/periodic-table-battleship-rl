# Protótipo GNN para a tabela periódica

O tabuleiro periódico não é uma grade densa: há lacunas e uma linha vazia entre
o corpo principal e o bloco f. Uma CNN precisa aprender que esses pixels não
representam células conectadas. Este protótipo representa diretamente a
topologia como grafo.

## Representação

- Cada uma das 118 células válidas é um nó.
- Arestas conectam somente vizinhos ortogonais legais definidos pela topologia.
- Cada nó recebe os canais públicos da observação e sua posição normalizada.
- A agregação inclui autoaresta e média dos vizinhos; portanto lacunas não
  recebem nem transmitem mensagens.
- A saída continua indexada no canvas de 180 ações para usar a mesma máscara de
  `AttackEnv` e permitir comparação com PPO/DQN.

O protótipo usa PyTorch puro, sem PyTorch Geometric ou dependência nova:

```powershell
uv run --extra train python scripts/run_gnn_periodic_prototype.py
```

O comando só verifica forma, conectividade e escolha legal de uma ação em um
estado público. Ele não treina, não compara desempenho e não produz resultado
para o README. A próxima etapa, se o DQN/CNN justificar a prioridade, é plugar
esta rede no protocolo cego de treino, validação e teste em uma issue própria.
