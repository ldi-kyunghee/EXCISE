# Experimental data preparation

Rebuild the MedQA-TD data used in the experiments. For the exact released files, [download the dataset](https://huggingface.co/datasets/yrhong/MedQA-TD).

[`medqa_test_order.json`](medqa_test_order.json) stores the test question order and the mapping between MedQA-TD IDs and original MedQA IDs.

## Requirements

Run from the repository root after [installation](../README.md#installation). All required packages are listed in [`requirements.txt`](../requirements.txt). Use a CUDA GPU with enough memory for GPT-OSS-120B.

## Build

Test:

```bash
python -m medqa_td.build \
    --type bystander --split test --fallback-mode fixed-template \
    --output outputs/bystander_test
```

Validation:

```bash
python -m medqa_td.build \
    --type nonliteral --split dev --fallback-mode none \
    --output outputs/nonliteral_dev
```

Replace `--type` with `bystander` or `nonliteral` for each run. Builder split `dev` corresponds to the released `validation` split.

To resume an interrupted run, rerun the same command with `--resume`, keeping the output directory and settings unchanged.

All options: `python -m medqa_td.build --help`.

## Outputs

| File | Contents |
|---|---|
| `medqa-td-<type>.jsonl` | Questions with selected distractors |
| `pairs.jsonl` | Original and selected-distractor versions of each question, with options, gold answer, and target option |
| `views.jsonl` | Original questions plus three distractor versions: selected, random, and random targeting the same option as the selected distractor |
| `selection.jsonl` | Which distractor was chosen for each question, or why the question was excluded |
| `summary.json` | Total numbers of questions and distractor candidates, including how many were kept |
| `.state/` | Saved progress for resuming an interrupted run |

Use `medqa-td-<type>.jsonl` as the `--data` input for EXCISE evaluation.

