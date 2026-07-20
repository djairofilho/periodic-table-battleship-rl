# Análise v0.3 por seed

A unidade de reamostragem é o seed cego. Observações repetidas de uma
mesma política ou dos cinco PPOs são reduzidas à média dentro do seed
antes do bootstrap. Os intervalos são percentis bootstrap bilaterais de
95%; eles descrevem estes seeds avaliados e não provam generalização além
do protocolo.

## Comparações pareadas

| Experimento | Cenário | Atacante/escopo | Comparação | Diferença | IC 95% | Leitura |
| --- | --- | --- | --- | ---: | ---: | --- |
| attack | battleship | — | MaskablePPO (multi-seed) − Hunt-target | +32.48 | [+29.43; +35.59] | favorece referência |
| attack | battleship | — | MaskablePPO (multi-seed) − Random masked | -1.14 | [-2.26; +0.04] | inconclusivo |
| attack | dense-118 | — | MaskablePPO (multi-seed) − Hunt-target | +39.98 | [+37.04; +43.00] | favorece referência |
| attack | dense-118 | — | MaskablePPO (multi-seed) − Random masked | -1.09 | [-2.15; +0.06] | inconclusivo |
| attack | periodic-table-battleship | — | MaskablePPO (multi-seed) − Hunt-target | +41.45 | [+37.93; +45.01] | favorece referência |
| attack | periodic-table-battleship | — | MaskablePPO (multi-seed) − Random masked | -1.52 | [-2.78; -0.22] | favorece candidata |
| placement | battleship | hunt-target-v1 | MaskablePPO placement (multi-seed) − dispersion-placement-v1 | -4.95 | [-7.79; -2.15] | favorece referência |
| placement | battleship | hunt-target-v1 | MaskablePPO placement (multi-seed) − hunt-target-resistant-placement-v1 | -5.52 | [-8.43; -2.65] | favorece referência |
| placement | battleship | hunt-target-v1 | MaskablePPO placement (multi-seed) − random-legal-placement-v1 | -3.32 | [-6.16; -0.58] | favorece referência |
| placement | battleship | maskable-ppo-v1:v03-attack-classic-s3103:v0.3-fixed-suite-seed-3103-step-20000 | MaskablePPO placement (multi-seed) − dispersion-placement-v1 | +2.35 | [+1.83; +2.94] | favorece candidata |
| placement | battleship | maskable-ppo-v1:v03-attack-classic-s3103:v0.3-fixed-suite-seed-3103-step-20000 | MaskablePPO placement (multi-seed) − hunt-target-resistant-placement-v1 | +10.42 | [+9.78; +11.11] | favorece candidata |
| placement | battleship | maskable-ppo-v1:v03-attack-classic-s3103:v0.3-fixed-suite-seed-3103-step-20000 | MaskablePPO placement (multi-seed) − random-legal-placement-v1 | +6.32 | [+5.30; +7.40] | favorece candidata |
| placement | battleship | random-masked-v1 | MaskablePPO placement (multi-seed) − dispersion-placement-v1 | +0.65 | [-0.41; +1.76] | inconclusivo |
| placement | battleship | random-masked-v1 | MaskablePPO placement (multi-seed) − hunt-target-resistant-placement-v1 | +0.33 | [-0.70; +1.40] | inconclusivo |
| placement | battleship | random-masked-v1 | MaskablePPO placement (multi-seed) − random-legal-placement-v1 | -0.69 | [-1.72; +0.45] | inconclusivo |
| placement | battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | MaskablePPO placement (multi-seed) − dispersion-placement-v1 | -2.09 | [-7.10; +2.95] | inconclusivo |
| placement | battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | MaskablePPO placement (multi-seed) − hunt-target-resistant-placement-v1 | -0.86 | [-5.38; +3.72] | inconclusivo |
| placement | battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | MaskablePPO placement (multi-seed) − random-legal-placement-v1 | -0.40 | [-5.60; +4.89] | inconclusivo |
| placement | periodic-table-battleship | hunt-target-v1 | MaskablePPO placement (multi-seed) − dispersion-placement-v1 | -0.87 | [-3.93; +2.21] | inconclusivo |
| placement | periodic-table-battleship | hunt-target-v1 | MaskablePPO placement (multi-seed) − hunt-target-resistant-placement-v1 | -4.83 | [-7.98; -1.65] | favorece referência |
| placement | periodic-table-battleship | hunt-target-v1 | MaskablePPO placement (multi-seed) − random-legal-placement-v1 | +2.66 | [-0.70; +5.99] | inconclusivo |
| placement | periodic-table-battleship | maskable-ppo-v1:v03-attack-periodic-s3101:v0.3-fixed-suite-seed-3101-step-30000 | MaskablePPO placement (multi-seed) − dispersion-placement-v1 | +5.66 | [+5.00; +6.33] | favorece candidata |
| placement | periodic-table-battleship | maskable-ppo-v1:v03-attack-periodic-s3101:v0.3-fixed-suite-seed-3101-step-30000 | MaskablePPO placement (multi-seed) − hunt-target-resistant-placement-v1 | +8.18 | [+7.26; +9.25] | favorece candidata |
| placement | periodic-table-battleship | maskable-ppo-v1:v03-attack-periodic-s3101:v0.3-fixed-suite-seed-3101-step-30000 | MaskablePPO placement (multi-seed) − random-legal-placement-v1 | +6.70 | [+5.51; +8.03] | favorece candidata |
| placement | periodic-table-battleship | random-masked-v1 | MaskablePPO placement (multi-seed) − dispersion-placement-v1 | -0.94 | [-2.10; +0.28] | inconclusivo |
| placement | periodic-table-battleship | random-masked-v1 | MaskablePPO placement (multi-seed) − hunt-target-resistant-placement-v1 | -0.68 | [-1.85; +0.54] | inconclusivo |
| placement | periodic-table-battleship | random-masked-v1 | MaskablePPO placement (multi-seed) − random-legal-placement-v1 | -1.97 | [-3.05; -0.84] | favorece referência |
| placement | periodic-table-battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | MaskablePPO placement (multi-seed) − dispersion-placement-v1 | +0.14 | [-5.63; +5.98] | inconclusivo |
| placement | periodic-table-battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | MaskablePPO placement (multi-seed) − hunt-target-resistant-placement-v1 | -0.43 | [-6.10; +5.35] | inconclusivo |
| placement | periodic-table-battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | MaskablePPO placement (multi-seed) − random-legal-placement-v1 | -1.64 | [-7.32; +4.05] | inconclusivo |

## Resumos por política

| Experimento | Cenário | Atacante/escopo | Política | Seeds | Episódios | Média | IC 95% |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| attack | battleship | — | Hunt-target | 100 | 100 | 61.75 | [58.68; 64.82] |
| attack | battleship | — | MaskablePPO (multi-seed) | 100 | 500 | 94.23 | [93.67; 94.76] |
| attack | battleship | — | Random masked | 100 | 100 | 95.37 | [94.29; 96.36] |
| attack | dense-118 | — | Hunt-target | 100 | 100 | 71.62 | [68.60; 74.64] |
| attack | dense-118 | — | MaskablePPO (multi-seed) | 100 | 500 | 111.60 | [111.11; 112.09] |
| attack | dense-118 | — | Random masked | 100 | 100 | 112.69 | [111.73; 113.58] |
| attack | periodic-table-battleship | — | Hunt-target | 100 | 100 | 69.33 | [65.80; 72.95] |
| attack | periodic-table-battleship | — | MaskablePPO (multi-seed) | 100 | 500 | 110.78 | [110.29; 111.24] |
| attack | periodic-table-battleship | — | Random masked | 100 | 100 | 112.30 | [111.02; 113.53] |
| placement | battleship | hunt-target-v1 | MaskablePPO placement (multi-seed) | 100 | 500 | 61.27 | [60.01; 62.50] |
| placement | battleship | hunt-target-v1 | dispersion-placement-v1 | 100 | 100 | 66.22 | [63.81; 68.61] |
| placement | battleship | hunt-target-v1 | hunt-target-resistant-placement-v1 | 100 | 100 | 66.79 | [64.35; 69.31] |
| placement | battleship | hunt-target-v1 | random-legal-placement-v1 | 100 | 100 | 64.59 | [62.00; 67.22] |
| placement | battleship | maskable-ppo-v1:v03-attack-classic-s3103:v0.3-fixed-suite-seed-3103-step-20000 | MaskablePPO placement (multi-seed) | 100 | 500 | 100.00 | [100.00; 100.00] |
| placement | battleship | maskable-ppo-v1:v03-attack-classic-s3103:v0.3-fixed-suite-seed-3103-step-20000 | dispersion-placement-v1 | 100 | 100 | 97.65 | [97.06; 98.17] |
| placement | battleship | maskable-ppo-v1:v03-attack-classic-s3103:v0.3-fixed-suite-seed-3103-step-20000 | hunt-target-resistant-placement-v1 | 100 | 100 | 89.58 | [88.89; 90.20] |
| placement | battleship | maskable-ppo-v1:v03-attack-classic-s3103:v0.3-fixed-suite-seed-3103-step-20000 | random-legal-placement-v1 | 100 | 100 | 93.68 | [92.60; 94.71] |
| placement | battleship | random-masked-v1 | MaskablePPO placement (multi-seed) | 100 | 500 | 95.35 | [94.87; 95.83] |
| placement | battleship | random-masked-v1 | dispersion-placement-v1 | 100 | 100 | 94.70 | [93.72; 95.64] |
| placement | battleship | random-masked-v1 | hunt-target-resistant-placement-v1 | 100 | 100 | 95.02 | [94.10; 95.88] |
| placement | battleship | random-masked-v1 | random-legal-placement-v1 | 100 | 100 | 96.04 | [95.04; 96.94] |
| placement | battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | MaskablePPO placement (multi-seed) | 100 | 500 | 83.59 | [79.92; 87.27] |
| placement | battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | dispersion-placement-v1 | 100 | 100 | 85.68 | [82.24; 88.89] |
| placement | battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | hunt-target-resistant-placement-v1 | 100 | 100 | 84.45 | [81.50; 87.23] |
| placement | battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | random-legal-placement-v1 | 100 | 100 | 83.99 | [80.48; 87.34] |
| placement | periodic-table-battleship | hunt-target-v1 | MaskablePPO placement (multi-seed) | 100 | 500 | 72.99 | [71.56; 74.46] |
| placement | periodic-table-battleship | hunt-target-v1 | dispersion-placement-v1 | 100 | 100 | 73.86 | [70.89; 77.04] |
| placement | periodic-table-battleship | hunt-target-v1 | hunt-target-resistant-placement-v1 | 100 | 100 | 77.82 | [74.86; 80.80] |
| placement | periodic-table-battleship | hunt-target-v1 | random-legal-placement-v1 | 100 | 100 | 70.33 | [67.31; 73.40] |
| placement | periodic-table-battleship | maskable-ppo-v1:v03-attack-periodic-s3101:v0.3-fixed-suite-seed-3101-step-30000 | MaskablePPO placement (multi-seed) | 100 | 500 | 118.00 | [118.00; 118.00] |
| placement | periodic-table-battleship | maskable-ppo-v1:v03-attack-periodic-s3101:v0.3-fixed-suite-seed-3101-step-30000 | dispersion-placement-v1 | 100 | 100 | 112.34 | [111.65; 113.00] |
| placement | periodic-table-battleship | maskable-ppo-v1:v03-attack-periodic-s3101:v0.3-fixed-suite-seed-3101-step-30000 | hunt-target-resistant-placement-v1 | 100 | 100 | 109.82 | [108.73; 110.76] |
| placement | periodic-table-battleship | maskable-ppo-v1:v03-attack-periodic-s3101:v0.3-fixed-suite-seed-3101-step-30000 | random-legal-placement-v1 | 100 | 100 | 111.30 | [109.96; 112.50] |
| placement | periodic-table-battleship | random-masked-v1 | MaskablePPO placement (multi-seed) | 100 | 500 | 111.97 | [111.32; 112.59] |
| placement | periodic-table-battleship | random-masked-v1 | dispersion-placement-v1 | 100 | 100 | 112.91 | [111.94; 113.83] |
| placement | periodic-table-battleship | random-masked-v1 | hunt-target-resistant-placement-v1 | 100 | 100 | 112.65 | [111.64; 113.63] |
| placement | periodic-table-battleship | random-masked-v1 | random-legal-placement-v1 | 100 | 100 | 113.94 | [113.02; 114.78] |
| placement | periodic-table-battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | MaskablePPO placement (multi-seed) | 100 | 500 | 99.37 | [95.21; 103.50] |
| placement | periodic-table-battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | dispersion-placement-v1 | 100 | 100 | 99.23 | [95.04; 103.25] |
| placement | periodic-table-battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | hunt-target-resistant-placement-v1 | 100 | 100 | 99.80 | [95.95; 103.39] |
| placement | periodic-table-battleship | v0.3-fixed-suite-random-hunt-frozen-ppo | random-legal-placement-v1 | 100 | 100 | 101.01 | [97.28; 104.54] |
