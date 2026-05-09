# Cross-Axis Correction

Decoupling jailbreak detection from correction in LLM activation space — see [docs/OVERVIEW.md](docs/OVERVIEW.md) for what this project is, the folder map, and how the capping mechanism works.

This README covers install and run only.

## Prerequisites

- Python 3.10+
- A CUDA GPU with enough memory for the target model (≥80 GB VRAM for Qwen3-32B; multiple GPUs for Llama-3.3-70B in bf16)
- `HF_TOKEN` set in the environment (HuggingFace), with access to gated models like Llama-3.3
- `ANTHROPIC_API_KEY` set in the environment (only for the LLM-judge step on benign outputs)

## Install

On a fresh machine, the bootstrap script handles HF CLI, hf_transfer, flash-attn, dependencies, and the package install in one go:

```bash
chmod +x scripts/setup/bootstrap.sh
./scripts/setup/bootstrap.sh
hf auth login            # paste your HF token when prompted
```

If you'd rather do it manually:

```bash
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install -e .         # installs the crosscap package so `python -m crosscap.…` works
```

## Run — Qwen3-32B (single GPU, no calibration step)

Qwen has a published assistant axis on HuggingFace, so you can skip calibration:

```bash
chmod +x scripts/launchers/run_qwen.sh
./scripts/launchers/run_qwen.sh sanity              # smoke test (5 prompts)
./scripts/launchers/run_qwen.sh full                # the real run (250 + 100 prompts)
```

Output lands in `results/crosscap_qwen_<preset>_<flags>/`. For bootstrap CIs:

```bash
python -m crosscap.analysis.bootstrap_results <output-dir>
```

## Run — Llama-3.3-70B end-to-end

Llama needs outcome-labelled calibration first because the published assistant axis isn't tuned for it:

```bash
# 1. Build the labelled calibration set (writes data/calibration/refusing.csv + compliant.csv)
./scripts/calibration/build_calibration.sh both 200 harmbench

# 2. Run the experiment (auto-uses data/calibration/ if it exists)
./scripts/launchers/run_llama.sh full                       # writes results/crosscap_llama_…/

# 3. Bootstrap CIs over the finished run
python -m crosscap.analysis.bootstrap_results results/crosscap_llama_…/
```

## Multi-GPU runs

`scripts/launchers/run_parallel.sh` runs the 3-stage pipeline (warmup → per-GPU chunks → merge) automatically:

```bash
./scripts/launchers/run_parallel.sh full 4    # 4-way data parallelism
./scripts/launchers/run_parallel.sh sanity 2  # smoke test on 2 GPUs
```

If you need finer control, the same three stages by hand:

```bash
python -m crosscap.core.run_crosscap --preset full --warmup
CUDA_VISIBLE_DEVICES=0 python -m crosscap.core.run_crosscap --preset full --chunk 0/4 &
CUDA_VISIBLE_DEVICES=1 python -m crosscap.core.run_crosscap --preset full --chunk 1/4 &
CUDA_VISIBLE_DEVICES=2 python -m crosscap.core.run_crosscap --preset full --chunk 2/4 &
CUDA_VISIBLE_DEVICES=3 python -m crosscap.core.run_crosscap --preset full --chunk 3/4 &
wait
python -m crosscap.core.run_crosscap --preset full --merge
```

## Sweeps & diagnostics

Exploratory scripts off the critical path — useful when you want to understand what a run is doing, not when you just want to reproduce a number.

| Script | What it sweeps |
|---|---|
| `scripts/sweeps/run_steer_probe.sh` | Direct steering on the compliance axis at fixed targets — does forcing the projection flip a benign prompt to refusal? |
| `scripts/sweeps/run_steer_layer_sweep.sh` | Layer ranges × steering targets — is the refusal basin a property of the late band only? |
| `scripts/sweeps/run_steer_layer_sweep_mixed.sh` | Same, on a mixed prompt set (5 benign + 5 jailbreak + 5 adversarial). |
| `scripts/sweeps/run_steer_layer_sweep_mixed_lowsweep.sh` | Same, but sweeping low targets (2, 4, 5, 7) to find the gentle-zone boundary. |
| `scripts/sweeps/run_steer_layer_sweep_jbb_compliance.sh` | JBB-only, steering toward the compliance pole (negative targets). Does it flip refused prompts to compliance? |
| `scripts/sweeps/run_steer_scopes.sh` | Same axes, four clamp scopes — `all`, `prefill_only`, `first_token_only`, `cursor_plus_first`. |
| `scripts/sweeps/run_sweep_detect.sh` | Sweeps absolute values of the assistant-axis detection threshold, with optional reclassification per τ. |

## Re-running analysis on an existing run

Point each tool at the run directory the launcher produced (e.g. `results/crosscap_qwen_full_…/`):

```bash
# 95% bootstrap CIs (no GPU needed — pure CSV math)
python -m crosscap.analysis.bootstrap_results <run-dir>

# Re-judge the outputs (optionally with a different judge model)
./scripts/analysis/run_reclassify.sh <run-dir>

# Per-layer axis diagnostics from a warmup.pt
python -m crosscap.analysis.diagnose_axes --warmup <run-dir>/warmup.pt
```

## License

Research use. See repository for details.
