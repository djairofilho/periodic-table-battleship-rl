# Roadmap 0.8–0.9

## Status atual

- `0.8` concluída: base de documentação pública, site com assets rastreáveis, badges no README e primeira trilha de release.
- `0.9` concluída: foco em validação científica e decisão de promoção da campanha.

### Fechamento explícito da v0.9

1. Microcalibração Bayesiana (`v0.9-bayes-sampler-calibration`) com `case_set=extended`.
2. Ablação de amostradores (`v0.9-bayes-sampler-ablation`) com custo/qualidade.
3. Validação multi-topologia em `smoke` (`v0.9-bayes-cross-topology-validation/smoke`).
4. Ampliação de demonstrações públicas (`v0.9-demonstrations`).
5. Treino/ablação CNN+GNN com 4 seeds de treino e 4 seeds de validação (`v0.9-bayesian-students`).
6. Relatório de fechamento (`docs/32-relatorio-v0.9.md`) e sincronização de assets.

## Roteiro técnico para 0.10 (condicional)

1. Definir nova campanha de ataque com 3 cenários e 5–10 seeds de validação por candidato.
2. Gate de promoção reforçado:
   - ganho em pelo menos 2 cenários com IC95% de melhora,
   - ausência de sinais de vazamento e queda de risco de truncação,
   - custo de treino avaliado com CPU/GPU.
3. Executar teste cego apenas quando o gate for aprovado.
4. Abrir self-play após candidato vencedor aprovado:
   - liga atacante-vs-atacante e atacante-vs-posicionador em suíte fixa.
5. Publicar versão 0.10 com novos gráficos, manifests e manifestos de dataset.

## Observações

- Self-play permanece adiado até haver candidato robusto.
- Todos os resultados negativos relevantes continuam rastreados no mesmo padrão de artefatos (`artifacts/` JSON/CSV/PNG/MD).
