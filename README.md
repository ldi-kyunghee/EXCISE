# EXCISE: Locating and Excising Distracting Sentences in Medical Question Answering

Paper: [arXiv:2610.35558](https://arxiv.org/abs/2610.35558)

## Installation

```bash
git clone https://github.com/ldi-kyunghee/EXCISE.git
cd EXCISE
conda create -n excise python=3.10 -y
conda activate excise
pip install -r requirements.txt
```

## Quick start

Run from the repository root. Download an evaluation file:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="yrhong/MedQA-TD",
    repo_type="dataset",
    filename="nonliteral/test.jsonl",
    local_dir="data/MedQA-TD",
)
```

Evaluate with and without EXCISE:

```bash
python -m excise.evaluate \
    --model qwen3_8b \
    --data data/MedQA-TD/nonliteral/test.jsonl \
    --out runs/qwen3_8b/nonliteral_test
```

Results are saved to `summary.json` and `predictions.jsonl` under `--out`.
For input formats, model settings, and evaluation options, see the [evaluation guide](excise/README.md).

## Reproduction guides

| Directory | Instructions |
|---|---|
| [`excise/`](excise/README.md) | Models, evaluation options, and outputs |
| [`medqa_td/`](medqa_td/README.md) | Rebuilding the experimental data |

Released data: [MedQA-TD](https://huggingface.co/datasets/yrhong/MedQA-TD). Rebuilding it is optional.

## Citation

```bibtex
@misc{hong2026exciselocatingexcisingdistracting,
      title={EXCISE: Locating and Excising Distracting Sentences in Medical Question Answering}, 
      author={Yerim Hong and Sohyun Son and Heeseung Chung and Mujeen Sung},
      year={2026},
      eprint={2610.35558},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.35558}, 
}
```

## License

Code: [MIT](LICENSE). For dataset licensing and source attribution, see the [MedQA-TD dataset card](https://huggingface.co/datasets/yrhong/MedQA-TD#license).
