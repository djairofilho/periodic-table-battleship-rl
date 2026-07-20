# Demonstração local e replays públicos

A demonstração de ataque é uma interface de terminal mínima sobre o mesmo
`AttackEnv` usado nos experimentos. Ela serve para inspecionar uma partida e
produzir um replay reproduzível, sem transformar a interface em uma fonte de
estado secreto.

## Executar uma política fixa

```powershell
uv run python -m periodic_table_battleship_rl.demo --topology periodic-table-battleship --seed 20260720 --policy hunt_target-v1 --replay-out runs/demo.json
```

O comando informa, tanto no terminal quanto no arquivo, a topologia, a seed e
a política. As políticas disponíveis são `random_masked-v1` e
`hunt_target-v1`.

## Jogar manualmente

```powershell
uv run python -m periodic_table_battleship_rl.demo --interactive --topology periodic-table-battleship --seed 20260720 --replay-out runs/jogada-manual.json
```

Digite um índice de ação de `0` a `179` ou uma coordenada zero-based no formato
`linha,coluna`, por exemplo `8,3`. A interface recusa lacunas e células já
chamadas. Use `quit` para interromper e salvar uma partida parcial.

O tabuleiro usa somente estes símbolos públicos:

| Símbolo | Significado |
| --- | --- |
| `·` | célula jogável ainda desconhecida |
| `H` | acerto em navio ainda não afundado |
| `S` | segmento de navio afundado |
| `o` | água |

Não há símbolo para navio não atingido. A demonstração também não revela a
frota ao terminar: assim a mesma regra vale durante a partida, no terminal e
no replay salvo.

## Verificar um replay

```powershell
uv run python -m periodic_table_battleship_rl.demo --replay runs/demo.json
```

A verificação recria o ambiente com a seed registrada e exige que cada ação e
resultado público coincida. Arquivos com campos fora do formato são recusados.

O JSON contém somente a versão do formato, topologia, seed, política, ações e
resultados públicos. Ele não contém `fleet`, posições ocupadas, identificadores
de navio ou quadros secretos. A seed permite reproduzir o episódio para fins de
auditoria, portanto o replay deve ser tratado como um artefato experimental,
não como um mecanismo de sigilo criptográfico.
