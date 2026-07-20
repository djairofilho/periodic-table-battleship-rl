# Transferência PPO entre topologias

Protocolo: `cross-topology-public-observation-v1`. Cada célula usa 100 seeds de teste fixos.
A diagonal é o controle same-topology; as demais células são transferência explícita.

| Treino | Teste | Média de tiros válidos | Taxa de vitória |
| --- | --- | ---: | ---: |
| battleship | battleship | 94.80 | 1.000 |
| battleship | dense-118 | 111.52 | 1.000 |
| battleship | periodic-table-battleship | 109.88 | 1.000 |
| dense-118 | battleship | 94.77 | 1.000 |
| dense-118 | dense-118 | 111.70 | 1.000 |
| dense-118 | periodic-table-battleship | 111.29 | 1.000 |
| periodic-table-battleship | battleship | 94.21 | 1.000 |
| periodic-table-battleship | dense-118 | 111.87 | 1.000 |
| periodic-table-battleship | periodic-table-battleship | 111.02 | 1.000 |
