# MAS Quadrapong

Course project for PKU 多智能体基础: 2v2 Quadrapong with IPPO, MAPPO, QMIX, and opponent-pool self-play.

Main materials live in `quadrapong/`:

- `src/`, `scripts/`: algorithms, environment wrapper, training, evaluation, plotting, analysis, and demo recording scripts.
- `report/HANDOFF.md`: full project handoff from model training.
- `report/DATA_ANALYSIS_AND_DEMO.md`: data-analysis and demo-recording handoff for the current work.
- `results/analysis/`: report-ready matchup CSV/Markdown summaries and heatmaps.
- `results/plots/`: learning curves and matchup trajectories.
- `results/videos/`: four recorded demo matches with JSON metadata.
- `checkpoints/*/*_final.pt`: RAM final checkpoints used by the demos and evaluation.

Pixel checkpoints and large TensorBoard/intermediate training artifacts are intentionally not committed to keep the repository cloneable. They remain on the remote training machine if needed.
