# Small-Model Alignment Starter

This branch is the student starter package for the SFT + DPO alignment assignment. It is intentionally minimal: it provides the tested environment metadata, assignment PDF, fixed evaluation/data files, and no completed training implementation.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.

---

# Small-Model Alignment: Implementation and Results (PA2)

**Author:** Shahzaib Ali (Roll No. 27100328)  
**Base Models:** `Qwen/Qwen2.5-0.5B` & `HuggingFaceTB/SmolLM2-360M-Instruct`  
**Judge Model:** `google/gemini-3-flash-preview` via Kaggle Model Proxy API  
**Overleaf Package:** `report_overleaf.zip` (contains `main.tex` and `figures/`)

---

## 1. Summary of Quantitative Results

Evaluations conducted under locked deterministic decoding ($rp=1.1$, greedy) via double-blind LLM-as-a-judge:

| Benchmark / Track | Candidate Model | Baseline Model | N | Wins | Losses | Ties | Win% (excl) | 95% Exact CI | $p$-value |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AlpacaEval 300** | **DPO-UF** | SFT | 300 | **133** | 86 | 81 | **60.73%** | [53.9%, 67.2%] | **0.0018** (sig.) |
| **HH-RLHF 300** | **DPO-HH** | SFT | 300 | **115** | 96 | 89 | **54.50%** | [47.5%, 61.4%] | 0.2152 (noise) |
| **AlpacaEval 300** | **SFT** | Base Qwen | 300 | **133** | 119 | 48 | **52.78%** | [46.4%, 59.1%] | 0.4129 |
| **HH-RLHF 300** | **SFT** | Base Qwen | 300 | **132** | 104 | 59 | **55.93%** | [49.3%, 62.4%] | 0.0786 |
| **Persona 24 (Overall)** | **3-Ep LoRA** | Base SmolLM2 | 24 | 10 | 11 | 3 | 47.62% | [25.7%, 70.2%] | 1.0000 |
| *(Style Dimension)* | **3-Ep LoRA** | Base SmolLM2 | 24 | **7** | 0 | 17 | **100.0%** | [59.0%, 100%] | **0.0156** (sig.) |
| **Persona 24 (Overall)** | **1-Ep LoRA** | Base SmolLM2 | 24 | **10** | 7 | 7 | **58.82%** | [32.9%, 81.6%] | 0.6291 |
| *(Style Dimension)* | **1-Ep LoRA** | Base SmolLM2 | 24 | 0 | 0 | 24 | — *(tied)* | — | 1.0000 |
| **Persona 24 (Overall)** | **3-Ep LoRA** | 1-Ep LoRA | 24 | 9 | 9 | 6 | 50.00% | [26.0%, 74.0%] | 1.0000 |
| *(Style Dimension)* | **3-Ep LoRA** | 1-Ep LoRA | 24 | **9** | 0 | 15 | **100.0%** | [66.4%, 100%] | **0.0039** (sig.) |

---

## 2. Report Table and Figure Mapping

Every table and figure in `report/main.tex` is linked directly to its generation script and source data files:

### Figures (300 DPI, `report/figures/`)
| Figure in Report | Source Data / Metric Log | Generating Script | Description |
| :--- | :--- | :--- | :--- |
| **Figure 1(a): SFT Loss** | `outputs/sft/metrics.jsonl` | `scripts/make_plots.py` | Training and validation cross-entropy loss on UltraChat 30k with best step (800) marked. |
| **Figure 1(b): DPO-HH Training** | `outputs/dpo_hh/metrics.jsonl` | `scripts/make_plots.py` | 3 panels: DPO loss, implicit reward accuracy, and reward margin with best step (400) marked. |
| **Figure 1(c): DPO-UF Training** | `outputs/dpo_uf/metrics.jsonl` | `scripts/make_plots.py` | 3 panels: DPO loss, implicit reward accuracy, and reward margin with best step (500) marked. |
| **Figure 1(d): LoRA Persona DPO** | `outputs/lora_persona/metrics.jsonl` | `scripts/make_plots.py` | 3 panels: LoRA DPO loss, reward accuracy, and reward margin across 1-ep and 3-ep runs. |

### Tables (Inlined in `report/main.tex` & `report/tables/`)
| Table in Report | Source File(s) in Codebase | Generating Script | Description |
| :--- | :--- | :--- | :--- |
| **Table 1: Configuration** | `configs/*.yaml`, training logs | `scripts/make_tables.py` | SFT, DPO-HH, DPO-UF, LoRA-DPO hyperparameters and locked generation settings. |
| **Table 2: Deviations Log** | `DEVIATIONS.md` | `scripts/make_tables.py` | Resource-driven vs Performance-driven deviations from initial manual guide. |
| **Table 3: Training Dynamics** | `outputs/*/metrics.jsonl` | `scripts/make_tables.py` | Best steps, validation losses, reward accuracies, and margins across all tracks. |
| **Table 4: Quantitative Results** | `outputs/judgments/*.jsonl` | `scripts/make_tables.py` | Wins, losses, ties, win rates, Clopper-Pearson 95% CIs, and binomial $p$-values. |
| **Table 5: Persona Dimensions** | `outputs/judgments/judge_persona_*.jsonl` | `scripts/make_tables.py` | Multi-dimensional breakdown: Style Faithfulness, Usefulness, and Safety. |
| **Table 6: Length Split Analysis** | `outputs/judgments/judge_alpaca_sft_vs_dpo_uf_rp11.jsonl` | `scripts/make_tables.py` | DPO-UF win rate when longer vs shorter than SFT, excluding 512-token hits. |
| **Table 7: Position Bias Check** | `outputs/judgments/judge_swap50_*.jsonl` | `scripts/make_tables.py` | Agreement, tie disagreements, direct flips, and overall position A preference. |
| **Table 8: Safety Audit Table** | `reports/persona_safety.md` | `scripts/make_tables.py` | All 6 safety-sensitive prompts: Base vs 1-Ep LoRA vs 3-Ep LoRA (rp=1.1). |
| **Table 9: Artifact Inventory** | Entire repository file structure | `scripts/make_tables.py` | Complete file path mapping for all configs, weights, generations, and judgments. |

---

## 3. How to Run and Reproduce

### Run Public Adapter Tests
```bash
python -m pytest tests/
```
All 21 unit tests (adapter smoke, numerical edge cases, and path resolution) will pass.

### Regenerate Tables and Figures
```bash
python scripts/make_plots.py
python scripts/make_tables.py
python scripts/build_final_latex_report.py
```

### Overleaf Compilation
The complete, self-contained Overleaf LaTeX bundle is packaged at `report_overleaf.zip`.
Importing `report_overleaf.zip` into Overleaf compiles cleanly under standard `pdflatex` (article class, 11pt, 1in margins, max 8 pages).
