# shared/evalution_pipeline

Shared modules for the uncertainty-aware inference evaluation pipeline.
All teams should use these instead of writing their own loaders or metrics.

## Modules

| File | What it does |
|------|--------------|
| `model_loader.py` | Load and free models — `load_model()`, `free_model()` |
| `data_loader.py` | Dataset configs — `DATASET_CONFIGS` (hellaswag, triviaqa, pubmedqa) |
| `eval_utils.py` | Run evaluation — `run_eval()`, calibration metrics, plots |
| `result_format.py` | Save results — called internally by `run_eval()` |

## Templates

| File | When to use |
|------|-------------|
| `eval_template.py` | Copy to your team folder and edit the TODOs |
| `eval_template_args.py` | Same, but config is passed via `--config` flag |

## Quick start

1. Copy a template into your team folder:
   ```bash
   cp shared/eval_template.py TeamB/run_eval.py
   ```

2. Fill in the TODOs — your `MODEL_REGISTRY` import, `CONFIG_KEY`, and `model_tag`.

3. Run:
   ```bash
   python TeamB/run_eval.py
   ```

See `TeamA/run_eval.py` and `TeamA/configs.py` for a working example.

## Output

Each dataset run saves four files under `<output_dir>/`:
```
<model_tag>_<quant_method>_<precision>_<dataset>.pt        # raw tensors
<model_tag>_<quant_method>_<precision>_<dataset>.json      # metrics summary
<model_tag>_<quant_method>_<precision>_<dataset>_reliability.png
<model_tag>_<quant_method>_<precision>_<dataset>_entropy.png
```

## Notes

- Keep `SEED = 42` for cross-team comparability.
- Some configs require a HuggingFace token (`HF_TOKEN=hf_...`) for gated Meta models.
