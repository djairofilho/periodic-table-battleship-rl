# Treinamentos CNN/GNN v0.9 (multi-seed)

- Arquiteturas: `['cnn', 'gnn']`
- Seeds de treino: `[15001, 15002, 15003, 15004]`
- Seeds de validação: `[16001, 16002, 16003, 16004]`

| Cenário | Hunt-target | Melhor estudante | Tiros válidos | AUC | Acordo (treino) |
| --- | ---: | --- | ---: | ---: | ---: |
| `battleship` | 63.75 | gnn-h16-s0.00 | 55.25 | 0.7319 | 0.067 |
| `dense-118` | 65.50 | cnn-h16-s0.50 | 79.00 | 0.7205 | 0.072 |
| `periodic-table-battleship` | 52.00 | gnn-h16-s0.00 | 65.50 | 0.7223 | 0.092 |
