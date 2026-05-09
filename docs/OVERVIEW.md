# Cross-Axis Capping — project overview

Reference doc for what this project is, how the codebase is laid out, and how the underlying capping mechanism works. For install and run instructions, see the top-level [README.md](../README.md).

## What this is

When an LLM generates text, its hidden states carry directional signals that indicate whether it's about to comply with or refuse a request. Standard "capping" interventions clamp these signals along a single direction (the **assistant axis**), using the same axis to both detect a jailbreak and correct the model's behavior.

This project asks: **what if we use two separate axes?**

- **Assistant-axis capping** — same direction detects and corrects.
- **Cross-axis capping** — detects on the assistant axis, but corrects along a separate **compliance axis** derived from PCA on the model's own refusing vs. compliant activations.

Tested on Qwen3-32B and Llama-3.3-70B-Instruct, cross-axis capping nearly doubles the jailbreak refusal rate (Qwen 32.9% → 54.5%; Llama 20.6% → 48.1%) while preserving 91–98% of benign output quality, with no additional compute. See [RESULTS.md](RESULTS.md) for the full numbers.

This project was done as part of the BlueDot Technical AI Safety project sprint. Compute costs were covered by Rapid Grants. Reach out at manuxtmail@gmail.com if anything in here is unclear.

## Folder map

```
cross-capping/
├── pyproject.toml             package metadata (declares src/ layout)
├── requirements.txt           Python dependencies
│
├── src/crosscap/              the Python package — all source code
│   ├── core/                    library + main orchestrator
│   ├── calibration/             builds the labeled refusing/compliant activation set
│   ├── analysis/                LLM-judge labelling + bootstrap CIs over a finished run
│   └── sweeps/                  diagnostic sweeps (layer ranges, thresholds, scopes)
│
├── scripts/                   bash wrappers — the entry points you actually invoke
│   ├── setup/                   one-time machine bootstrap (HF CLI, flash-attn, deps)
│   ├── launchers/               model-specific runners (run_qwen.sh, run_llama.sh, …)
│   ├── calibration/             two-stage outcome-labelled calibration pipeline
│   ├── analysis/                rerun the LLM judge on an existing run dir
│   └── sweeps/                  diagnostic-sweep wrappers
│
├── docs/                      long-form documentation
│   ├── OVERVIEW.md              this file
│   ├── RESULTS.md               headline numbers + per-run config + interpretation
│   └── analysis_plan.md         5-step diagnostic framework for interpreting a run
│
├── notebooks/
│   └── primer.ipynb             walkthrough — first stop for a new reader
│
└── data/
    └── calibration/             outcome-labelled refusing.csv + compliant.csv (Llama input)
```

### Why each folder exists

| Folder | Role |
|---|---|
| `src/crosscap/core/` | The library (`crosscap_experiment.py` — model loading, capping hooks, axis math) and the main orchestrator (`run_crosscap.py` — the WARMUP → GENERATION → MERGE pipeline). Everything else imports from here. |
| `src/crosscap/calibration/` | Builds the labeled refusing/compliant activation CSVs that the cross-axis math depends on. Run once per model when the default JBB+WildJailbreak axes don't apply (Llama uses outcome-labelled calibration). |
| `src/crosscap/analysis/` | Post-experiment evaluation: LLM-judge labelling (`reclassify_refusals`) and bootstrap CIs (`bootstrap_results`) computed over an existing run directory. |
| `src/crosscap/sweeps/` | Diagnostic sweeps — over layer ranges, detection thresholds, steering targets, and clamp scopes. Off the critical path; used to interpret a run, not to produce one. |
| `scripts/setup/` | `bootstrap.sh` — one-time machine setup on a fresh box (HuggingFace CLI, hf_transfer, flash-attn, package install). |
| `scripts/launchers/` | The "press a button to reproduce a paper run" entry points: `run_qwen.sh`, `run_llama.sh`, `run_gemma.sh`, plus `run_parallel.sh` for multi-GPU and `download_model.sh` for cache pre-warming. |
| `scripts/calibration/` | `build_calibration.sh` — two-stage wrapper that runs `generate_calibration` then `classify_calibration` to produce `data/calibration/`. |
| `scripts/analysis/` | `run_reclassify.sh` — re-runs the LLM judge on an existing results directory (e.g., when the original run skipped reclassify, or you want a different judge model). |
| `scripts/sweeps/` | Bash wrappers for the sweep modules in `src/crosscap/sweeps/`, mostly for setting common flags and parameter grids. |
| `docs/` | This file, `RESULTS.md` (headline numbers + per-run config + bootstrap CIs), and `analysis_plan.md` (5-step diagnostic framework). |
| `notebooks/` | `primer.ipynb` — annotated walkthrough; recommended first stop. |
| `data/calibration/` | Outcome-labelled `refusing.csv` + `compliant.csv` consumed by `core/run_crosscap.py` to fit Llama's compliance axis. |

Run outputs (CSVs + `warmup.pt` + `metadata.json`) land in `results/<run-name>/` by default — created on demand by the launchers; pass `--output-dir` to put them elsewhere. `crosscap.analysis.bootstrap_results` and `scripts/analysis/run_reclassify.sh` operate on whatever path you point them at.

## How capping works (briefly)

During generation, the model builds a hidden state `h` at each layer. Capping modifies `h` in-place:

**Single-axis (assistant) capping** — same axis `v` for detection and correction:

```
h ← h − v · min(dot(h, v) − τ, 0)
```

**Cross-axis capping** — detect on `v`, correct along a separate compliance axis `c`:

```
if dot(h, v) < τ_detect:
    h ← h − c · min(dot(h, c) − τ_c, 0)
```

The compliance axis is built by PCA on hidden states the model produces when it refuses harmful requests vs. when it complies.

## Datasets

All pulled from HuggingFace at warmup time:

| Dataset | Role |
|---|---|
| [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | Bare harmful goals → "refusing" activations |
| [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) | Adversarial jailbreak prompts for eval, plus "compliant" activations from the train split |
| [assistant-axis-vectors](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors) | Pre-computed assistant axis (layer 53) for Qwen |

## Output CSV format

Each row is one prompt. Key columns:

| Column | Description |
|---|---|
| `prompt_text` | The input prompt |
| `baseline_text` | Uncapped output (control) |
| `{method}_cap_applied` | Did capping fire? (Yes / No) |
| `{method}_cap_layers` | Which layers fired (e.g. `L46,L47,L48`) |
| `{method}_cap_text` | Output with capping active |
| `llm_label` | Judge's classification (added by `reclassify_refusals`) |

## Where to read next

- [RESULTS.md](RESULTS.md) — full results with bootstrap CIs and per-run interpretation.
- [analysis_plan.md](analysis_plan.md) — five-step diagnostic framework (axis alignment, calibration shape, magnitude vs. direction, per-token trajectories, clamp persistence).
- [`../notebooks/primer.ipynb`](../notebooks/primer.ipynb) — annotated walkthrough.
