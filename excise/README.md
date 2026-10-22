# EXCISE evaluation

For [installation](../README.md#installation) and a complete execution example, see the [Quick start](../README.md#quick-start). Run all commands from the repository root.

## Input

The default input is four-option data. The final sentence of `question` is treated as the question stem.
Each JSONL row needs `id` (unique), `question`, `options` (`A`–`D`), and `answer` (gold letter).
Released [MedQA-TD files](https://huggingface.co/datasets/yrhong/MedQA-TD) can be used directly.

## Models

`--model` accepts a model key below or a compatible Hugging Face model ID. The listed models use their fixed checkpoints and settings; other models use untuned defaults (`layer=16`, `tau=1.0`, `gamma=0`).

| `--model` | Hugging Face model | Layer | `tau` | `gamma` |
|---|---|---:|---:|---:|
| `llama31_8b_inst` | `meta-llama/Llama-3.1-8B-Instruct` | 9 | 1.0 | 0 |
| `qwen3_8b` | `Qwen/Qwen3-8B` (thinking off) | 25 | 1.0 | 0 |
| `meerkat_8b` | `dmis-lab/llama-3-meerkat-8b-v1.0` | 19 | 0.5 | 0 |

## Arguments and options

Each run reports the selected model's base accuracy and its accuracy with EXCISE.

| Argument | What to provide |
|---|---|
| `--model` | A model key from the table above or a compatible Hugging Face model ID, e.g., `qwen3_8b` or `Qwen/Qwen3-8B` |
| `--data` | Path to the evaluation JSONL file in the input format above |
| `--out` | Directory for results and checkpoints; created if it does not exist |

By default, the evaluator appends a fixed JSON response prefix with an empty `rationale`, then reads the A–D/E logits immediately after `"answer": "`. The highest-scoring letter is the predicted answer; no explanation is generated.

| Option | Description |
|---|---|
| `--ablation` | Also report accuracy for EXCISE w/o IA (`no_ia`) and w/o DG (`no_dg`), using A–D/E logits at the same JSON `answer` position |
| `--generate` | Generate a rationale and final answer, then extract the answer from that response instead of directly reading answer-letter logits |

All options: `python -m excise.evaluate --help`.

## Outputs

| File | Contents |
|---|---|
| `summary.json` | Accuracy and intervention rate, in percent |
| `predictions.jsonl` | Answers and zero-based blocked sentence indices (`null` for no intervention) |
| `measured.jsonl` | Per-question option logits and decision margins; per-sentence attribution scores and answers after blocking each sentence |
| `vectors.npz` | Hidden-state vectors of each sentence and the remaining narrative, used by the Discordance Gate |
| `generations.jsonl` | Generated responses when using `--generate` |
