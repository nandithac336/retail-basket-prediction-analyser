# askThili requirement-to-evidence checklist

## Required milestones

| Requirement | Implementation | Evidence generated |
|---|---|---|
| Synthetic baskets with planted itemsets | `app/core.py`: `milestone_1()` | `data/baskets.csv`, `data/planted_itemsets.json` |
| N-gram next-item model | `NGramModel`, `milestone_2()` | Trained during `python -m app.cli run` |
| Small language model trained on basket sequences | `TinyNeuralLanguageModel`, `milestone_3()` | `outputs/basket_language_model.npz`, `outputs/vocabulary.json`, `outputs/training_loss.json` |
| Evaluate both on held-out data | `milestone_4()` | `outputs/metrics.json` and `outputs/comparison_report.md` |
| Compare and visualize results | `milestone_5()` | `outputs/model_comparison.png`, `outputs/training_loss.png`, report |

## Run-it checklist

- Python and Git installed.
- Create and activate `.venv`; install `requirements.txt`.
- Copy `.env.example` to `.env` and choose one Ollama/Groq/Gemini provider for the optional `app/llm.py` chat helper. Do not commit `.env`.
- `python -m app.cli hello` should print the ready message.
- `python -m pytest -q` should pass.
- `python -m app.cli run` should execute all five milestones.
- README documents setup, commands, artifacts, metrics, and limitations.

## Submission checks

- Public GitHub repository: `https://github.com/pruthvi-p06/retail-basket-prediction-analyzer`
- GitHub Actions: workflow should show a green check after push.
- askThili Ship tab: submit the public repository URL and inspect its automated review.

## Transparent caveats

- The neural language model is a compact feed-forward model trained from scratch; it is not SmolLM2-135M or a Transformer.
- Basket item order is shuffled to create synthetic next-item sequences; it is not a real chronological transaction dataset.
- The askThili starter asks for a commit after each milestone. The initial Git commit in this repository bundled all five completed milestones, so the original incremental-commit history was not preserved. Do not manufacture historical commits; make clear, meaningful commits for any future changes.
- Stretch ideas (different N-gram orders, basket-length slices, planted-itemset slices, error analysis) are optional, not required milestones.
