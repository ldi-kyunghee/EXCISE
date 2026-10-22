"""Build MedQA-TD (Section 3 of the paper): generate, validate, score and select Bystander or Nonliteral distractors.

    python -m medqa_td.build --type bystander --split test --output outputs/bystander

1. candidates: GPT-OSS-120B (vLLM) writes eight candidate distractors per wrong option and judges, under two option
   orders, whether each points to its intended option and not to the correct one; rejected candidates are regenerated
   for up to --retry-rounds extra rounds.
2. scoring: the same model, as the reference, answers the question with each accepted candidate inserted before the
   question sentence (A-D log-probabilities at the final-answer position; Transformers with MXFP4 weights).
3. selection: per question, prefer candidates that turn the reference model's correct answer wrong, then the largest
   margin of a wrong option over the gold option. With --fallback-mode fixed-template, questions without any accepted
   candidate get a fixed-template distractor (the test release uses this; the validation release excludes them).
Writes medqa-td-<type>.jsonl (the release format: id, question, options, answer, distractor, target, clean_question,
fixed_template) plus pairs, views, selection audit and summary. Generation and scoring run in separate processes so
that they can use different Python environments (--generation-python with vLLM, --scoring-python with MXFP4 support).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


# ------------------------------------------------------------------ Shared helpers: hashes, JSONL, inserting a sentence before the question sentence

CHOICES = ("A", "B", "C", "D")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def insert_before_final_sentence(question: str, distractor: str) -> str:
    """Exact MedDistractQA insertion parity used by the legacy pipeline."""
    import re

    sentences = re.split(r"(?<=[.!?]) +", question.strip())
    sentences.insert(-1, distractor.strip())
    return " ".join(sentences)


# ------------------------------------------------------------------ Atomic files and resumable checkpoints

def sha(value):
    if isinstance(value, Path): value = value.read_bytes()
    elif not isinstance(value, bytes): value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    os.replace(temp, path)


def sample_by_hash(pool, key, salt):
    if not pool: raise ValueError('Empty sampling pool')
    return min(pool, key=lambda c: sha([salt, key, c['candidate_id']]))

class Checkpoints:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def cached(self, key, inputs, action):
        path = self.root / (sha(key) + '.json')
        input_hash = sha(inputs)
        if path.exists():
            value = read(path)
            if value['key'] != key or value['input_hash'] != input_hash:
                raise ValueError('Checkpoint input drift: ' + key)
            if value['data_hash'] != sha(value['data']):
                raise ValueError('Checkpoint data corruption: ' + key)
            return value['data']
        data = action()
        atomic(path, dict(key=key, input_hash=input_hash, data=data, data_hash=sha(data)))
        return data


class Run:
    """Hold a single-writer lock, bind inputs/config/code, verify completed stages."""
    def __init__(self, output, binding, resume=False):
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()) and not resume:
            raise ValueError('Output is not empty; use --resume or a new output directory')
        self.state = self.output / '.state'
        self.state.mkdir(parents=True, exist_ok=True)
        self.lock = (self.state / 'run.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            root = Path(__file__).resolve().parent
            code = {p.name: sha(p) for p in sorted(root.glob('*')) if p.suffix in ('.py', '.json')}
            Checkpoints(self.state).cached('run', dict(binding=binding, code=code), lambda: True)
        except Exception:
            self.lock.close()
            raise

    def stage(self, name, action):
        marker = self.state / (name + '.json')
        if marker.exists():
            value = read(marker)
            for rel, digest in value['files'].items():
                path = self.output / rel
                if not path.is_file() or sha(path) != digest:
                    raise ValueError('Completed output changed: ' + rel)
            return
        paths = action()
        atomic(marker, {'files': {str(Path(p).relative_to(self.output)): sha(Path(p)) for p in paths}})

    def __enter__(self): return self
    def __exit__(self, *args): self.lock.close()


def child_environment(**overrides):
    root=str(Path(__file__).resolve().parents[1])
    env=dict(os.environ,**overrides)
    env['PYTHONPATH']=root+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def require(condition, message):
    if not condition:
        raise ValueError(message)


# ------------------------------------------------------------------ MedQA (GBaker/MedQA-USMLE-4-options-hf, pinned)


MEDQA_REPO = 'GBaker/MedQA-USMLE-4-options-hf'
MEDQA_REVISION = '17af9355ef89fba60de966eabaeba797c695f86e'
MEDQA_FILES = {split: split + '.json' for split in ('train', 'dev', 'test')}
TEST_ORDER = Path(__file__).resolve().parent / 'medqa_test_order.json'   # MedQA-TD test ids (test-00000, ...) -> HF row ids


def medqa_source(split):
    require(split in MEDQA_FILES, 'MedQA split must be train, dev, or test')
    return dict(repo_id=MEDQA_REPO, revision=MEDQA_REVISION, filename=MEDQA_FILES[split], split=split)


def content_key(question, options, answer):
    # Exact text and option order preserve the original inference prompts.
    return question, tuple(options[c] for c in 'ABCD'), answer


def normalize_rows(rows, split):
    require(isinstance(rows, list) and bool(rows), 'MedQA data must be nonempty')
    medqa_source(split)
    required = {'id', 'sent1', 'sent2', 'label'} | {f'ending{i}' for i in range(4)}
    result, ids, content = [], set(), set()
    for position, row in enumerate(rows):
        require(isinstance(row, dict) and required <= row.keys(), f'Invalid MedQA fields at row {position}')
        sid = row['id']
        match = re.fullmatch(split + r'-(\d+)', sid) if isinstance(sid, str) else None
        require(match is not None and sid not in ids, f'Invalid or duplicate MedQA ID at row {position}')
        require(isinstance(row['sent1'], str) and row['sent1'].strip(), f'Empty MedQA question: {sid}')
        require(row['sent2'] == '', f'Unexpected additional MedQA question text: {sid}')
        require(type(row['label']) is int and 0 <= row['label'] < 4, f'Invalid MedQA label: {sid}')
        options = {c: row[f'ending{i}'] for i, c in enumerate('ABCD')}
        require(all(isinstance(v, str) and v.strip() for v in options.values()), f'Invalid MedQA options: {sid}')
        answer = 'ABCD'[row['label']]
        key = content_key(row['sent1'], options, answer)
        require(key not in content, f'Duplicate MedQA question/options/answer: {sid}')
        ids.add(sid)
        content.add(key)
        result.append(dict(idx=int(match.group(1)), source_id=sid, question=row['sent1'],
                           options=options, answer_idx=answer))
    return result


def construction_order(rows, order):
    """Restore the existing test source IDs and iteration order without changing text."""
    by_id = {r['source_id']: r for r in rows}
    require(len(order) == len(rows) and {r['hf_id'] for r in order} == set(by_id)
            and {r['idx'] for r in order} == set(range(len(rows))), 'Invalid MedQA test source mapping')
    return [dict(by_id[r['hf_id']], idx=r['idx'], source_id=f"test-{r['idx']:05d}") for r in order]


def load_medqa(split='test', *, construction=False):
    from huggingface_hub import hf_hub_download
    source = medqa_source(split)
    path = Path(hf_hub_download(repo_id=source['repo_id'], repo_type='dataset',
        filename=source['filename'], revision=source['revision'], token=False))
    rows = normalize_rows(read_jsonl(path), split)
    metadata = dict(source, file_sha256=sha(path), available_questions=len(rows))
    if construction and split == 'test':
        order = read(TEST_ORDER)
        rows = construction_order(rows, order)
        metadata['alignment'] = [dict(source_id=r['source_id'], hf_id=o['hf_id']) for r, o in zip(rows, order)]
    return rows, metadata


# ------------------------------------------------------------------ Reference-model prompt (scoring)

SYSTEM_PROMPT = (
    "You are a medical AI assistant used for decision-making in a clinical setting."
)

USER_PROMPT_TEMPLATE = """Answer the following multiple-choice medical question.

Return only one compact JSON object in exactly this form: {{\"answer\":\"A\"}}
The answer must be one of A, B, C, or D. Do not provide reasoning, analysis, confidence, or any additional text.

Question:
{question}

Options:
A) {A}
B) {B}
C) {C}
D) {D}
"""

# This is appended after a complete user turn.  It bypasses Harmony's analysis
# channel and fixes the measurement at the first A-D answer token.
HARMONY_FINAL_ANSWER_PREFIX = (
    '<|start|>assistant<|channel|>final<|message|>{"answer":"'
)


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    options = row["options"]
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=row["question"],
        A=options["A"],
        B=options["B"],
        C=options["C"],
        D=options["D"],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ------------------------------------------------------------------ Generation prompts (Bystander, Nonliteral) and the fixed template

prompt_gen_beta_confounder = (
    '\n'
    'Example sentences:\n'
    '\n'
    '1. "The patient\'s sister said her friend has high blood pressure and has been having seizures."\n'
    '2. "The patient\'s dog has diabetes and cannot see properly."\n'
    '3. "The patient\'s neighbor mentioned that their cat has been sneezing and refusing to eat."\n'
    '4. "The patient\'s cousin said her coworker\'s father has arthritis and struggles to walk long distances."\n'
    '5. "The patient\'s brother mentioned that his friend\'s goldfish has been swimming erratically and losing color."\n'
    '    \n'
    '\n'
    '## Medical Question: {question}\n'
    '\n'
    '## Clinical topic: {clinical_topic}\n'
    '\n'
    'Generate one statement, where another person/animal is referenced with relation to a medical term as seen in the example sentences. The statement should also be related to the clinical topic. The statement should NOT affect the diagnostic process or decision-making for the medical question in any meaningful way. It should not convey any medical information or symptomatic information. Avoid including any information that could mislead clinical reasoning.\n'
    '\n'
    'Only respond with the confounding sentence.\n'
)

NONLITERAL_PROMPT = '\nExample sentences (for other questions):\n\n1. "The patient\'s zodiac sign is Cancer."\n2. "The patient said their friend\'s business is hemorrhaging money after a bad investment."\n3. "The patient mentioned that their new car has a transmission fluid leak."\n4. "The patient described their favorite art exhibit as having vascular pattern in the sculptures."\n5. "The patient\'s professor delivered a lecture about the heart of ancient civilizations."\n\n## Medical Question: {question}\n\n## Clinical topic: {clinical_topic}\n\nGenerate one sentence using a medical term related to the clinical topic, where clinical terminology is used as nonmedical meaning. The statement should NOT affect the diagnostic process or decision-making for the medical question in any meaningful way. It should not convey any medical information or symptomatic information. Avoid including any information that could mislead clinical reasoning.\n\nAlways begin with “The patient said that” or “The patient joked that” and use a clinical term figuratively to describe a nonmedical subject—not the reverse.\n\nDo not generate the following examples or similar statements that assert physiological effects or patient symptoms, bodily states, or test findings, even as jokes:\n- “The patient joked that the office coffee was so strong it could boost LH levels faster than a morning workout.” This asserts a physiological effect on LH levels.\n- “The patient joked that her coffee’s ‘Rinne test’ was always abnormal, because the espresso sounded louder in the left ear than the right.” This introduces a difference in the patient\'s hearing between the two ears.\n\nOnly respond with the confounding sentence.\n'


FIXED_FALLBACK_TEMPLATES = {
    'bystander': (
        'The patient\'s aunt said that her neighbor\'s parrot often repeated '
        'the phrase “{clinical_topic}”.'
    ),
    'nonliteral': (
        'The patient joked that “{clinical_topic}” would be a fitting '
        'nickname for their current mood.'
    ),
}

FIXED_FALLBACK_TEMPLATE_VERSIONS = {
    'bystander': 'bystander_parrot_v1',
    'nonliteral': 'nonliteral_mood_v1',
}


def render_fixed_fallback(kind, clinical_topic):
    if kind not in FIXED_FALLBACK_TEMPLATES:
        raise ValueError('Unknown fallback distractor type')
    if not isinstance(clinical_topic, str) or not clinical_topic.strip():
        raise ValueError('Fallback clinical topic must be nonempty')
    return (FIXED_FALLBACK_TEMPLATES[kind].format(clinical_topic=clinical_topic.strip()),
            FIXED_FALLBACK_TEMPLATE_VERSIONS[kind])


# ------------------------------------------------------------------ Target check under two option orders

JUDGE_PROMPT = """You are labeling which answer-choice concept is evoked by an unrelated bystander sentence in a medical multiple-choice question.

Medical question:
{question}

Answer choices (the display order may be shuffled; preserve the printed labels):
{choices}

Bystander sentence:
{sentence}

Independently determine which answer-choice concept the bystander sentence most strongly evokes. Do not solve the medical question. If exactly one option is evoked, use its printed label. Use NONE if no option is meaningfully evoked and MULTI if multiple options are comparably evoked.

Return only a JSON object with exactly these fields:
{{"primary_evoked_option":"A|B|C|D|NONE|MULTI","evoked_options":["A|B|C|D"],"relation_type":"LEXICAL_DIRECT|MORPHOLOGICAL|SEMANTIC|NONE|MULTI","strength":"strong|weak|none"}}
"""


def final_content(raw):
    marker = "<|channel|>final<|message|>"
    text = raw.rsplit(marker, 1)[-1] if marker in raw else raw
    for ending in ("<|return|>", "<|end|>", "<|call|>"):
        text = text.split(ending, 1)[0]
    return text.strip()


def parse_judgment(raw):
    text = final_content(raw)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None, "JSON_NOT_FOUND"
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return None, f"JSON_DECODE:{error.msg}"
    required = {"primary_evoked_option", "evoked_options", "relation_type", "strength"}
    if set(value) != required:
        return None, "SCHEMA_KEYS"
    if value["primary_evoked_option"] not in {"A", "B", "C", "D", "NONE", "MULTI"}:
        return None, "PRIMARY_ENUM"
    if not isinstance(value["evoked_options"], list) or any(
        option not in {"A", "B", "C", "D"} for option in value["evoked_options"]
    ):
        return None, "EVOKED_OPTIONS"
    if value["relation_type"] not in {
        "LEXICAL_DIRECT", "MORPHOLOGICAL", "SEMANTIC", "NONE", "MULTI"
    }:
        return None, "RELATION_ENUM"
    if value["strength"] not in {"strong", "weak", "none"}:
        return None, "STRENGTH_ENUM"
    return value, None


def pass_structure(
    parsed: dict[str, Any] | None, intended_target: str, gold: str
) -> str:
    if parsed is None:
        return "PARSE_INVALID"
    if parsed["strength"] == "none":
        return "NO_STRENGTH"
    if parsed["primary_evoked_option"] == "NONE" or parsed["relation_type"] == "NONE":
        return "NONE_TARGET"
    evidence = set(parsed["evoked_options"])
    primary = parsed["primary_evoked_option"]
    if primary in CHOICES:
        evidence.add(primary)
    wrong_options = {option for option in evidence if option != gold}
    if gold in evidence or primary == gold:
        return "GOLD_IN_MULTI" if wrong_options else "GOLD_TARGET"
    if (
        primary == intended_target
        and parsed["relation_type"] != "MULTI"
        and wrong_options == {intended_target}
    ):
        return "INTENDED_SINGLE_TARGET"
    if primary == "MULTI" or parsed["relation_type"] == "MULTI" or len(wrong_options) >= 2:
        return "MULTI_TARGET_WRONG_ONLY"
    return "OTHER"


def target_gate_result(
    judgment: dict[str, Any], intended_target: str, gold: str
) -> dict[str, Any]:
    forward = pass_structure(judgment["forward"]["parsed"], intended_target, gold)
    reverse = pass_structure(judgment["reverse"]["parsed"], intended_target, gold)
    structures = {forward, reverse}
    if structures == {"INTENDED_SINGLE_TARGET"}:
        return {
            "pass": True,
            "label": "INTENDED_SINGLE_TARGET",
            "forward_structure": forward,
            "reverse_structure": reverse,
        }
    allowed = {"INTENDED_SINGLE_TARGET", "MULTI_TARGET_WRONG_ONLY"}
    if structures.issubset(allowed) and "MULTI_TARGET_WRONG_ONLY" in structures:
        return {
            "pass": True,
            "label": "GOLD_FREE_MULTI",
            "forward_structure": forward,
            "reverse_structure": reverse,
        }
    return {
        "pass": False,
        "label": "REJECTED",
        "forward_structure": forward,
        "reverse_structure": reverse,
    }


def extract_final(raw):
    marker = "<|channel|>final<|message|>"
    text = raw.rsplit(marker, 1)[-1] if marker in raw else raw
    for ending in ("<|return|>", "<|end|>", "<|call|>"):
        text = text.split(ending, 1)[0]
    text = text.strip().strip('"').strip()
    return re.sub(r"^(?:confounding sentence|sentence)\s*:\s*", "", text, flags=re.I).strip()


# ------------------------------------------------------------------ Reference-model scoring (Transformers, MXFP4)

DEFAULT_MODEL = 'openai/gpt-oss-120b'
DEFAULT_REVISION = 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'


def as_token_ids(value: Any) -> list[int]:
    # Newer/custom Transformers chat templates may return a BatchEncoding
    # rather than the bare list returned by older releases.
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(
                "tokenizer mapping output does not contain an input_ids field"
            )
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError(f"unsupported tokenizer output type: {type(value).__name__}")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("expected exactly one tokenized prompt")
        value = value[0]
    return [int(token_id) for token_id in value]


def longest_common_prefix(sequences: list[list[int]]) -> list[int]:
    if not sequences:
        return []
    limit = min(len(sequence) for sequence in sequences)
    index = 0
    while index < limit and len({sequence[index] for sequence in sequences}) == 1:
        index += 1
    return sequences[0][:index]


def build_final_answer_token_plan(tokenizer) -> dict[str, Any]:
    sequences = {
        choice: as_token_ids(
            tokenizer.encode(
                HARMONY_FINAL_ANSWER_PREFIX + choice,
                add_special_tokens=False,
            )
        )
        for choice in CHOICES
    }
    common_prefix = longest_common_prefix(list(sequences.values()))
    candidate_segments = {
        choice: sequence[len(common_prefix) :]
        for choice, sequence in sequences.items()
    }
    if not common_prefix:
        raise RuntimeError("Harmony final answer prefix tokenized to an empty prefix")
    if any(len(segment) != 1 for segment in candidate_segments.values()):
        raise RuntimeError(
            f"A-D are not single-token alternatives after the final prefix: {candidate_segments}"
        )
    candidate_token_ids = {
        choice: segment[0] for choice, segment in candidate_segments.items()
    }
    if len(set(candidate_token_ids.values())) != 4:
        raise RuntimeError(f"A-D token ids are not unique: {candidate_token_ids}")
    decoded_prefix = tokenizer.decode(common_prefix, skip_special_tokens=False)
    required_markers = (
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
    )
    if any(marker not in decoded_prefix for marker in required_markers):
        raise RuntimeError(
            "tokenized prefix does not preserve the forced Harmony final channel: "
            + repr(decoded_prefix)
        )
    if "<|channel|>analysis" in decoded_prefix:
        raise RuntimeError("forced answer prefix unexpectedly enters the analysis channel")
    return {
        "candidate_token_ids": candidate_token_ids,
        "common_prefix_ids": common_prefix,
        "decoded_common_prefix": decoded_prefix,
        "probe_token_ids": sequences,
    }


def render_user_turn_tokens(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "reasoning_effort": "low",
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        effort = kwargs.pop("reasoning_effort")
        rendered = tokenizer.apply_chat_template(
            messages,
            chat_template_kwargs={"reasoning_effort": effort},
            **kwargs,
        )
    tokens = as_token_ids(rendered)
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    if "<|channel|>analysis" in decoded:
        raise RuntimeError(
            "base user-turn rendering already contains an assistant analysis channel"
        )
    return tokens


def normalize(logprobs: dict[str, float]) -> dict[str, float]:
    maximum = max(logprobs.values())
    weights = {choice: math.exp(value - maximum) for choice, value in logprobs.items()}
    denominator = sum(weights.values())
    return {choice: weights[choice] / denominator for choice in CHOICES}


def unique_top1_ties(logprobs: dict[str, float]) -> tuple[str | None, list[str]]:
    maximum = max(logprobs.values())
    winners = [choice for choice in CHOICES if logprobs[choice] == maximum]
    return (winners[0] if len(winners) == 1 else None), winners


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def score_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure GPT-OSS forced A-D preference with a Transformers direct "
            "forward pass, without vLLM generation or scheduling."
        )
    )
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--attention-implementation",
        default="eager",
        choices=("sdpa", "eager"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if "gpt-oss-120b" not in args.model.lower():
        parser.error("this diagnostic is restricted to GPT-OSS-120B")
    if args.max_model_len <= 1:
        parser.error("max-model-len must be greater than one")
    return args


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def last_token_logits(model, input_ids):
    """Compute only the final-position LM-head logits when the API allows it."""
    try:
        output = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        logits = output.logits
        if logits.ndim != 3 or logits.shape[1] != 1:
            raise RuntimeError(f"unexpected logits_to_keep output shape: {tuple(logits.shape)}")
        return logits[0, -1]
    except TypeError as exc:
        # Older compatible Transformers versions may not expose
        # logits_to_keep. Avoid materializing vocabulary logits for every
        # prompt position by running the backbone and applying lm_head only to
        # the final hidden state.
        if "logits_to_keep" not in str(exc):
            raise
        backbone = getattr(model, "model", None)
        lm_head = getattr(model, "lm_head", None)
        if backbone is None or lm_head is None:
            raise RuntimeError(
                "Transformers lacks logits_to_keep and the model does not expose model/lm_head"
            ) from exc
        hidden = backbone(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return lm_head(hidden[:, -1:, :])[0, -1]


def score_main(argv=None) -> None:
    args = score_args(argv)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before Python starts"
        )
    try:
        import accelerate
        import kernels
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import is_kernels_available, is_triton_available
    except ImportError as exc:
        raise RuntimeError(
            "Transformers direct MXFP4 scoring requires torch, transformers, "
            "accelerate, and kernels"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPT-OSS-120B direct scoring")
    if args.attention_implementation == "sdpa":
        raise RuntimeError(
            "this Transformers build does not support SDPA for GptOssForCausalLM; "
            "use --attention-implementation eager"
        )

    kernels_version = importlib.metadata.version("kernels")
    triton_version = importlib.metadata.version("triton")
    if not is_kernels_available():
        import transformers.utils.import_utils as import_utils

        minimum = getattr(import_utils, "KERNELS_MIN_VERSION", None)
        maximum = getattr(import_utils, "KERNELS_MAX_VERSION", None)
        raise RuntimeError(
            "Transformers does not accept the installed kernels package for MXFP4: "
            f"installed={kernels_version}, allowed_min={minimum}, "
            f"allowed_max={maximum}. Refusing the automatic BF16 dequantization "
            "because GPT-OSS-120B will not fit on one 96GB GPU."
        )
    if not is_triton_available("3.4.0"):
        raise RuntimeError(
            "MXFP4 requires Triton >=3.4.0: "
            f"installed={triton_version}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    request_path = Path(args.requests)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    requests = read_jsonl(request_path)
    request_ids = [row["request_id"] for row in requests]
    if not requests:
        raise ValueError("request file is empty")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("duplicate request ids")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    token_plan = build_final_answer_token_plan(tokenizer)
    rendered_requests: list[tuple[dict[str, Any], list[int], str]] = []
    for row in requests:
        prompt_tokens = (
            render_user_turn_tokens(tokenizer, build_messages(row))
            + token_plan["common_prefix_ids"]
        )
        if len(prompt_tokens) + 1 > args.max_model_len:
            raise RuntimeError(
                f"request exceeds max-model-len: {row['request_id']} {len(prompt_tokens)}"
            )
        rendered_requests.append(
            (
                row,
                prompt_tokens,
                sha256_text(",".join(map(str, prompt_tokens))),
            )
        )

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    quantization_config = getattr(config, "quantization_config", None) or {}
    if quantization_config.get("quant_method") != "mxfp4":
        raise RuntimeError(
            "expected the pinned GPT-OSS-120B checkpoint to declare MXFP4 quantization"
        )
    print(
        json.dumps(
            {
                "status": "cpu_prompt_preflight_complete_before_model_load",
                "requests": len(requests),
                "choice_token_ids": token_plan["candidate_token_ids"],
                "maximum_prompt_tokens": max(
                    len(prompt_tokens) for _, prompt_tokens, _ in rendered_requests
                ),
                "attention_implementation": args.attention_implementation,
                "deterministic_algorithms": True,
            },
            indent=2,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        dtype="auto",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation=args.attention_implementation,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    candidate_ids = token_plan["candidate_token_ids"]

    if args.overwrite:
        records: dict[str, dict[str, Any]] = {}
    elif output_path.exists():
        existing_rows = read_jsonl(output_path)
        records = {row["request_id"]: row for row in existing_rows}
        if len(records) != len(existing_rows):
            raise ValueError("duplicate request ids in existing output")
    else:
        records = {}

    expected_ids = set(request_ids)
    if not set(records) <= expected_ids:
        raise ValueError('Foreign request IDs in saved scores')
    for row, _, prompt_hash in rendered_requests:
        saved = records.get(row['request_id'], {})
        if saved.get('status') != 'ok': continue
        if (saved.get('rendered_prompt_token_sha256') != prompt_hash
                or saved.get('model_revision') != args.model_revision
                or saved.get('model_id') != args.model
                or any(saved.get(k) != row.get(k) for k in ('source_id','candidate_id','gold','intended_target','role','condition'))):
            raise ValueError('Saved scoring input or model differs: ' + row['request_id'])
        values = saved.get('option_logprobs', {})
        if set(values) != set(CHOICES) or not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Invalid cached option scores')
        if saved.get('top1') != unique_top1_ties(values)[0]:
            raise ValueError('Cached top1 does not match option scores')

    for index, (row, prompt_tokens, prompt_hash) in enumerate(rendered_requests, start=1):
        if records.get(row["request_id"], {}).get("status") == "ok":
            continue
        try:
            input_ids = torch.tensor(
                [prompt_tokens],
                dtype=torch.long,
                device=input_device,
            )
            with torch.inference_mode():
                logits = last_token_logits(model, input_ids)
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                option_logprobs = {
                    choice: float(logprobs[token_id].item())
                    for choice, token_id in candidate_ids.items()
                }
            if not all(math.isfinite(value) for value in option_logprobs.values()):
                raise ValueError("non-finite A-D logprob")
            top1, tied = unique_top1_ties(option_logprobs)
            generated_choice = top1 if top1 is not None else sorted(tied)[0]
            record = {
                "analysis_generated": False,
                "candidate_id": row.get("candidate_id"),
                "choice_score_valid": True,
                "condition": row["condition"],
                "finish_reason": "direct_forward_forced_choice",
                "forced_channel": "final",
                "generated_choice": generated_choice,
                "generated_token_id": candidate_ids[generated_choice],
                "gold": row["gold"],
                "intended_target": row.get("intended_target"),
                "model_id": args.model,
                "model_revision": args.model_revision,
                "option_choice_scores": normalize(option_logprobs),
                "option_logprobs": option_logprobs,
                "prompt_token_count": len(prompt_tokens),
                "rendered_prompt_token_sha256": prompt_hash,
                "request_id": row["request_id"],
                "role": row["role"],
                "source_id": row["source_id"],
                "source_idx": row["source_idx"],
                "status": "ok",
                "top1": top1,
                "top1_tied_choices": tied if len(tied) > 1 else [],
            }
        except Exception as exc:
            record = {
                "request_id": row["request_id"],
                "source_idx": row["source_idx"],
                "role": row["role"],
                "candidate_id": row.get("candidate_id"),
                "status": "request_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        records[row["request_id"]] = record
        atomic_write_jsonl(
            output_path,
            [records[request_id] for request_id in request_ids if request_id in records],
        )
        if index == 1 or index % 10 == 0 or index == len(rendered_requests):
            print(f"Transformers direct scoring: {index}/{len(rendered_requests)}")

    final_rows = [records[request_id] for request_id in request_ids if request_id in records]
    ok = sum(row.get("status") == "ok" for row in final_rows)
    errors = len(requests) - ok
    manifest = {
        "schema_version": "gpt_oss_transformers_direct_score_manifest_v1",
        "status": "complete" if errors == 0 else "technical_failure",
        "counts": {"requests": len(requests), "ok": ok, "errors": errors},
        "measurement": {
            "description": "post-prompt pre-rationale forced A-D preference",
            "forced_channel": "harmony_final",
            "analysis_generated": False,
            "allowed_choices": list(CHOICES),
            "logprob_retrieval": "direct_last_token_log_softmax",
            "runtime_mode": "transformers_direct",
            "batch_size": 1,
            "temperature": 0.0,
            "max_tokens": 1,
            "seed": args.seed,
        },
        "determinism": {
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "two_independent_processes_required_for_repeatability_audit": True,
        },
        "model": {
            "backend": "transformers_direct_forward",
            "model_id": args.model,
            "model_revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision,
            "quantization": "mxfp4",
            "attention_implementation": args.attention_implementation,
            "torch_version": getattr(torch, "__version__", None),
            "transformers_version": getattr(transformers, "__version__", None),
            "accelerate_version": getattr(accelerate, "__version__", None),
            "kernels_version": kernels_version,
            "triton_version": triton_version,
            "model_dtype": str(getattr(model, "dtype", None)),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        },
        "prompt": {
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
            "user_prompt_template_sha256": sha256_text(USER_PROMPT_TEMPLATE),
            "harmony_final_answer_prefix_sha256": sha256_text(
                HARMONY_FINAL_ANSWER_PREFIX
            ),
            "choice_token_ids": candidate_ids,
            "common_prefix_ids": token_plan["common_prefix_ids"],
        },
        "bindings": {
            "requests": {"path": str(request_path), "sha256": sha256_file(request_path)},
            "outputs": {"path": str(output_path), "sha256": sha256_file(output_path)},
            "scorer": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
            "forced_choice_helpers": {
                "path": str(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "must_not_be_used_for_training": True,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"Scoring complete: {ok}/{len(requests)}")
    if errors:
        raise RuntimeError(f"Transformers direct scoring failed for {errors} requests")


# ------------------------------------------------------------------ Selection policy

@dataclass(frozen=True)


class CandidateDecision:
    status: str
    selected_candidate_id: str | None
    selected_by_flip_priority: bool | None
    tied_candidate_ids: tuple[str, ...]


def unique_top1(option_logprobs: dict[str, float]) -> str | None:
    maximum = max(option_logprobs.values())
    winners = [choice for choice in CHOICES if option_logprobs[choice] == maximum]
    return winners[0] if len(winners) == 1 else None


def overall_difficulty(option_logprobs: dict[str, float], gold: str) -> float:
    return max(option_logprobs[choice] for choice in CHOICES if choice != gold) - option_logprobs[gold]


def enrich_candidate(
    candidate: dict[str, Any], clean_top1: str | None, gold: str
) -> dict[str, Any]:
    logprobs = candidate["option_logprobs"]
    candidate_top1 = unique_top1(logprobs)
    difficulty = overall_difficulty(logprobs, gold)
    harmful_flip = clean_top1 == gold and candidate_top1 in CHOICES and candidate_top1 != gold
    return {
        **candidate,
        "candidate_top1": candidate_top1,
        "overall_difficulty": difficulty,
        "construction_harmful_flip": harmful_flip,
    }


def decide_candidate(
    candidates: list[dict[str, Any]], clean_top1: str | None, gold: str
) -> tuple[CandidateDecision, list[dict[str, Any]]]:
    if not candidates:
        return (
            CandidateDecision("no_gate_valid_candidate", None, None, ()),
            [],
        )
    enriched = [enrich_candidate(candidate, clean_top1, gold) for candidate in candidates]
    flip_candidates = [row for row in enriched if row["construction_harmful_flip"]]
    pool = flip_candidates if flip_candidates else enriched
    maximum = max(row["overall_difficulty"] for row in pool)
    winners = [row for row in pool if row["overall_difficulty"] == maximum]
    winners.sort(key=lambda row: row["candidate_id"])
    if len(winners) != 1:
        return (
            CandidateDecision(
                "unresolved_exact_D_tie",
                None,
                bool(flip_candidates),
                tuple(row["candidate_id"] for row in winners),
            ),
            enriched,
        )
    winner = winners[0]
    status = (
        "selected_from_flip_candidates"
        if flip_candidates
        else "selected_by_max_D_fallback"
    )
    return (
        CandidateDecision(
            status,
            winner["candidate_id"],
            bool(flip_candidates),
            (),
        ),
        enriched,
    )

RESIDUAL_HASH_SALT = 'gpt-oss-hard-v2-residual-tie-v1'


@dataclass(frozen=True)


class ResidualCandidateDecision:
    final_status: str
    selected_candidate_id: str | None
    selected_by_flip_priority: bool | None
    base_selection_status: str
    resolution_method: str
    tied_candidate_ids: tuple[str, ...]
    canonical_candidate_ids: tuple[str, ...]
    discarded_duplicate_candidate_ids: tuple[str, ...]
    selected_tie_break_hash: str | None


def deterministic_tie_hash(
    candidate: dict[str, Any],
    source_id: str,
    *,
    salt: str = RESIDUAL_HASH_SALT,
) -> str:
    fields = (
        salt,
        source_id,
        str(candidate["candidate_id"]),
        str(candidate["candidate_text_sha256"]),
        str(candidate["intended_target"]),
        str(candidate["target_gate_label"]),
    )
    return sha256("|".join(fields).encode("utf-8")).hexdigest()


def _duplicate_identity(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate["candidate_text_sha256"]),
        str(candidate["intended_target"]),
        str(candidate["target_gate_label"]),
    )


def _canonical_duplicate_representatives(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        for required in (
            "candidate_id",
            "candidate_rank",
            "candidate_text_sha256",
            "intended_target",
            "target_gate_label",
        ):
            if required not in candidate:
                raise KeyError(f"residual selector requires {required}")
        groups.setdefault(_duplicate_identity(candidate), []).append(candidate)

    representatives: list[dict[str, Any]] = []
    discarded: list[str] = []
    for rows in groups.values():
        ordered = sorted(
            rows,
            key=lambda row: (int(row["candidate_rank"]), str(row["candidate_id"])),
        )
        representatives.append(ordered[0])
        discarded.extend(str(row["candidate_id"]) for row in ordered[1:])
    representatives.sort(key=lambda row: str(row["candidate_id"]))
    discarded.sort()
    return representatives, discarded


def decide_candidate_with_residual_policy(
    candidates: list[dict[str, Any]],
    clean_top1: str | None,
    gold: str,
    source_id: str,
) -> tuple[ResidualCandidateDecision, list[dict[str, Any]]]:
    base, enriched = decide_candidate(candidates, clean_top1, gold)

    if base.selected_candidate_id is not None:
        return (
            ResidualCandidateDecision(
                final_status="selected",
                selected_candidate_id=base.selected_candidate_id,
                selected_by_flip_priority=base.selected_by_flip_priority,
                base_selection_status=base.status,
                resolution_method="primary_rule_unique_winner",
                tied_candidate_ids=(),
                canonical_candidate_ids=(),
                discarded_duplicate_candidate_ids=(),
                selected_tie_break_hash=None,
            ),
            enriched,
        )

    if base.status == "no_gate_valid_candidate":
        return (
            ResidualCandidateDecision(
                final_status="excluded_construction_infeasible",
                selected_candidate_id=None,
                selected_by_flip_priority=None,
                base_selection_status=base.status,
                resolution_method="no_gate_valid_exclusion",
                tied_candidate_ids=(),
                canonical_candidate_ids=(),
                discarded_duplicate_candidate_ids=(),
                selected_tie_break_hash=None,
            ),
            enriched,
        )

    if base.status != "unresolved_exact_D_tie":
        raise ValueError(f"unsupported residual selector state: {base.status}")

    enriched_index = {str(row["candidate_id"]): row for row in enriched}
    tied = [enriched_index[candidate_id] for candidate_id in base.tied_candidate_ids]
    representatives, discarded = _canonical_duplicate_representatives(tied)
    canonical_ids = tuple(str(row["candidate_id"]) for row in representatives)

    if len(representatives) == 1:
        winner = representatives[0]
        method = "duplicate_text_canonical_provenance"
        selected_hash = None
    else:
        hashed = [
            (deterministic_tie_hash(row, source_id), str(row["candidate_id"]), row)
            for row in representatives
        ]
        selected_hash, _, winner = min(hashed, key=lambda value: (value[0], value[1]))
        method = "distinct_candidate_deterministic_hash"

    return (
        ResidualCandidateDecision(
            final_status="selected",
            selected_candidate_id=str(winner["candidate_id"]),
            selected_by_flip_priority=base.selected_by_flip_priority,
            base_selection_status=base.status,
            resolution_method=method,
            tied_candidate_ids=base.tied_candidate_ids,
            canonical_candidate_ids=canonical_ids,
            discarded_duplicate_candidate_ids=tuple(discarded),
            selected_tie_break_hash=selected_hash,
        ),
        enriched,
    )


# ------------------------------------------------------------------ Candidate generation with retries

@dataclass(frozen=True)
class RetryPolicy:
    slots: int = 8
    extra_rounds: int = 2
    base_seed: int = 42
    namespace: str = 'nl-cue-retry-v2'

    def __post_init__(self):
        if not 1 <= self.slots <= 8 or not 0 <= self.extra_rounds <= 2:
            raise ValueError('Supported budget: 1..8 slots, 0..2 extra rounds')
        if not 0 <= self.base_seed < 2**31 or not self.namespace:
            raise ValueError('Invalid seed or namespace')


def round_seed(policy, source_id, target, round_index):
    if not 0 <= round_index <= policy.extra_rounds:
        raise ValueError('Round outside approved budget')
    seeds = [policy.base_seed]
    for index in range(1, round_index + 1):
        seed = int(sha([policy.namespace, policy.base_seed, source_id, target, index])[:16], 16) % (2**31)
        while seed in seeds:
            seed = (seed + 1) % (2**31)
        seeds.append(seed)
    return seeds[-1]


def candidate_id(policy, source_id, target, slot, round_index):
    return f'{policy.namespace}:{source_id}:{target}:s{slot}:r{round_index}'


def classify_failure(candidate, judgments, target, gold, gate):
    """Use only the bystander option-cue decision after technical validation."""
    if (not candidate.get('sentence', '').strip()
            or '<|channel|>' in candidate['sentence']
            or candidate.get('finish_reason') != 'stop'):
        return dict(category='technical', retryable=False, reasons=['GENERATION_INCOMPLETE_OR_EMPTY'], gate=None)
    technical = []
    for order in ('forward', 'reverse'):
        item = judgments.get(order, {})
        if item.get('parsed') is None or item.get('error') or item.get('finish_reason') != 'stop':
            technical.append(order + ':JUDGMENT_INCOMPLETE_OR_INVALID')
    if technical:
        return dict(category='technical', retryable=False, reasons=technical, gate=None)
    result = gate(judgments, target, gold)
    if result['pass']:
        return dict(category='accepted', retryable=False, reasons=[], gate=result)
    allowed = {'NO_STRENGTH', 'GOLD_IN_MULTI', 'GOLD_TARGET', 'NONE_TARGET', 'OTHER',
               'INTENDED_SINGLE_TARGET', 'MULTI_TARGET_WRONG_ONLY'}
    structures = [result['forward_structure'], result['reverse_structure']]
    if not set(structures) <= allowed:
        raise ValueError(f'Unclassified gate failure: {structures}')
    return dict(category='option_cue', retryable=True, reasons=structures, gate=result)


def run_target(source, target, generation_template, judge_template, policy,
               checkpoint_dir, generate, judge, gate, runtime_binding,
               check=lambda: None, progress=lambda event: None):
    """Callbacks: generate(prompt,n,seed)->list; judge(prompt)->parsed+raw dict.

    No correctness, confidence, difficulty, or score enters the retry decision.
    Each callback must preserve raw output and finish_reason. Schema validation
    of judge output belongs to the unchanged legacy parser used by the backend.
    """
    if target not in 'ABCD' or target == source['answer_idx']:
        raise ValueError('Target must be an original wrong choice')
    if set(source['options']) != set('ABCD'):
        raise ValueError('Expected original A-D options')
    store = Checkpoints(checkpoint_dir)
    sid = source['source_id']
    job = sid + ':' + target
    lock = (store.root / (sha(job) + '.lock')).open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(source=source, target=target, generation_template=generation_template,
                       judge_template=judge_template, policy=asdict(policy), runtime=runtime_binding,
                       gate_contract='bystander_option_cue_v1_no_content_gate')
        store.cached('binding:' + job, binding, lambda: binding)
        prompt = generation_template.format(question=source['question'], clinical_topic=source['options'][target])
        active = list(range(policy.slots))
        attempts, accepted, slots = [], [], {}
        for round_index in range(policy.extra_rounds + 1):
            if not active:
                break
            check()
            seed = round_seed(policy, sid, target, round_index)
            inputs = dict(binding=sha(binding), prompt=prompt, n=len(active), slots=active, seed=seed)
            batch = store.cached(f'generate:{job}:r{round_index}', inputs,
                                 lambda: generate(prompt, len(active), seed))
            if len(batch) != len(active):
                raise ValueError('Generation returned the wrong batch size')
            again = []
            for slot, output in zip(active, batch):
                check()
                cid = candidate_id(policy, sid, target, slot, round_index)
                can = dict(output, candidate_id=cid, source_id=sid, source_idx=source['idx'],
                           intended_target=target, slot=slot, round=round_index,
                           candidate_rank=slot * (policy.extra_rounds + 1) + round_index,
                           seed=seed, prompt_sha256=sha(prompt))
                can['candidate_text_sha256'] = hashlib.sha256(can['sentence'].encode()).hexdigest()
                judgments = {}
                if can['sentence'].strip() and can['finish_reason'] == 'stop' and '<|channel|>' not in can['sentence']:
                    for order, tag in [('ABCD', 'forward'), ('DCBA', 'reverse')]:
                        check()
                        jprompt = judge_template.format(question=source['question'], sentence=can['sentence'],
                            choices='\n'.join(f'{k}. {source["options"][k]}' for k in order))
                        judgments[tag] = store.cached(f'judge:{cid}:{order}',
                            dict(binding=sha(binding), prompt=jprompt, seed=policy.base_seed),
                            lambda p=jprompt: judge(p))
                decision = classify_failure(can, judgments, target, source['answer_idx'], gate)
                record = dict(candidate=can, judgments=judgments, decision=decision)
                saved = store.cached('decision:' + cid, record, lambda r=record: r)
                attempts.append(saved)
                status = decision['category']
                if status == 'accepted':
                    accepted.append(dict(can, generation_valid=True, judgments=judgments,
                                         target_gate_label=decision['gate']['label'],
                                         candidate_origin='generated_gate_valid',
                                         gate_validated=True, selection_eligible=True))
                elif decision['retryable'] and round_index < policy.extra_rounds:
                    again.append(slot)
                elif decision['retryable']:
                    status = 'option_cue_budget_exhausted'
                slots[slot] = dict(slot=slot, final_candidate_id=cid, status=status,
                                   attempts=round_index + 1)
            progress(dict(source_id=sid, target=target, round=round_index, generated=len(batch),
                          accepted_total=len(accepted), next_round_n=len(again)))
            active = again
        result = dict(source_id=sid, target=target, policy=asdict(policy),
                      attempts=attempts, valid_candidates=accepted,
                      slots=[slots[s] for s in range(policy.slots)], generated_n=len(attempts),
                      accepted_n=len(accepted), shortage_n=policy.slots-len(accepted))
        if len(attempts) > policy.slots * (policy.extra_rounds + 1) or len(accepted) > policy.slots:
            raise ValueError('Retry budget violated')
        store.cached('complete:' + job, result, lambda: result)
        return result
    finally:
        lock.close()


def fixed_fallback_candidates(source, condition):
    """Create one explicit, non-gate-validated fallback for each wrong option."""
    candidates = []
    for order, target in enumerate(choice for choice in 'ABCD' if choice != source['answer_idx']):
        sentence, version = render_fixed_fallback(condition, source['options'][target])
        candidate_id = f'fixed-template-fallback-v1:{condition}:{source["source_id"]}:{target}'
        candidates.append(dict(candidate_id=candidate_id, source_id=source['source_id'],
            source_idx=source['idx'], intended_target=target, candidate_rank=10000 + order,
            sentence=sentence, candidate_text_sha256=hashlib.sha256(sentence.encode()).hexdigest(),
            candidate_origin='fixed_template_fallback', generation_valid=False,
            gate_validated=False, selection_eligible=True, judgments={},
            target_gate_label='FIXED_TEMPLATE_FALLBACK', target_set=[target],
            target_set_source='template_binding', fallback_template_version=version,
            fallback_trigger='no_gate_valid_candidate_after_retry_budget',
            finish_reason='deterministic_template'))
    return candidates


def build_scoring_pool(sources, results, insert, sample, condition="nonliteral",
                       random_salt="nl-random-valid-v1", fallback_mode='none'):
    """The SAME frozen accepted pool feeds score requests and Random/Hard."""
    if fallback_mode not in ('none', 'fixed-template'):
        raise ValueError('Unknown fallback mode')
    index = {r['source_id']: r for r in sources}
    if len(index) != len(sources):
        raise ValueError('Duplicate source IDs')
    jobs = [(r['source_id'], r['target']) for r in results]
    if len(set(jobs)) != len(jobs):
        raise ValueError('Duplicate source-target results')
    valid, requests, random = [], [], {}
    def request(row, question, can=None):
        return dict(request_id=can['candidate_id'] if can else 'clean:'+row['source_id'],
                    source_id=row['source_id'], source_idx=row['idx'], role='candidate' if can else 'clean',
                    condition=condition if can else 'clean', question=question, options=row['options'],
                    gold=row['answer_idx'], candidate_id=can['candidate_id'] if can else None,
                    intended_target=can['intended_target'] if can else None,
                    candidate_origin=can.get('candidate_origin') if can else None,
                    gate_validated=can.get('gate_validated') if can else None)
    for source in sources:
        requests.append(request(source, source['question']))
    for result in results:
        row = index[result['source_id']]
        for can in result['valid_candidates']:
            if can['source_id'] != row['source_id'] or can['intended_target'] != result['target']:
                raise ValueError('Candidate provenance mismatch')
            valid.append(can)
    generated_source_ids = {can['source_id'] for can in valid}
    fallback_source_ids = []
    if fallback_mode == 'fixed-template':
        for source in sources:
            if source['source_id'] not in generated_source_ids:
                fallback_source_ids.append(source['source_id'])
                valid.extend(fixed_fallback_candidates(source, condition))
    for can in valid:
        row = index[can['source_id']]
        requests.append(request(row, insert(row['question'], can['sentence']), can))
    if len({r['candidate_id'] for r in valid}) != len(valid):
        raise ValueError('Duplicate accepted candidate IDs')
    for sid in index:
        pool = [r for r in valid if r['source_id'] == sid]
        if pool:
            random[sid] = sample(pool, sid, random_salt)['candidate_id']
    return dict(valid_candidates=valid, score_requests=requests, random_selection=random,
                generated_source_coverage_n=len(generated_source_ids),
                fallback_source_ids=fallback_source_ids,
                fallback_source_n=len(fallback_source_ids),
                fallback_candidate_n=sum(c.get('candidate_origin') == 'fixed_template_fallback' for c in valid),
                source_coverage_n=len(random), source_n=len(sources), fallback_mode=fallback_mode)


# ------------------------------------------------------------------ Selected questions and release rows

def candidate_target_set(candidate):
    if candidate is None:
        return []
    if candidate.get('target_set_source') == 'template_binding':
        return list(candidate['target_set'])
    return sorted(set(sum([candidate['judgments'][tag]['parsed']['evoked_options']
                           for tag in ('forward', 'reverse')], [])))

def validate_scores(requests, scores):
    ids = [r['request_id'] for r in scores]
    if len(set(ids)) != len(ids) or set(ids) != {r['request_id'] for r in requests} or len(scores) != len(requests):
        raise ValueError('Score coverage or duplicate mismatch')
    for row in scores:
        values = row.get('option_logprobs', {})
        if (row.get('status') != 'ok' or row.get('model_revision') != DEFAULT_REVISION
                or set(values) != set('ABCD') or not all(math.isfinite(v) for v in values.values())):
            raise ValueError('Invalid model score')
    return {r['request_id']: r for r in scores}

def select_dataset(sources, pool, scores, namespace="nl-dev890-v1", random_salt="nl-random-same-target-v1"):
    index = validate_scores(pool['score_requests'], scores)
    grouped = {s['source_id']: [] for s in sources}
    for can in pool['valid_candidates']:
        grouped[can['source_id']].append(dict(can, option_logprobs=index[can['candidate_id']]['option_logprobs']))
    output, audits = [], []
    for source in sources:
        sid, gold = source['source_id'], source['answer_idx']
        candidates = grouped[sid]
        decision, _ = decide_candidate_with_residual_policy(candidates, index['clean:'+sid]['top1'], gold, sid)
        audits.append(dict(source_id=sid, **decision.__dict__))
        if decision.selected_candidate_id is None:
            continue
        by_id = {r['candidate_id']:r for r in candidates}
        hard = by_id[decision.selected_candidate_id]
        chosen = dict(hard=hard, random_valid=by_id[pool['random_selection'][sid]],
            random_same_target=sample_by_hash([r for r in candidates if r['intended_target']==hard['intended_target']],
                                                sid, random_salt))
        same_target_control_collapsed = (chosen['random_same_target']['candidate_id'] == hard['candidate_id'])
        for view in ('clean','random_valid','hard','random_same_target'):
            can = chosen.get(view)
            origin = can.get('candidate_origin', 'generated_gate_valid') if can else None
            row = dict(request_id=f'{namespace}:{sid}:{view}', source_id=sid, source_idx=source['idx'],
                view=view, question=insert_before_final_sentence(source['question'],can['sentence']) if can else source['question'],
                clean_question=source['question'], options=source['options'], gold_answer=gold,
                intended_target=can['intended_target'] if can else hard['intended_target'],
                candidate_id=can['candidate_id'] if can else None,
                target_gate_label=can['target_gate_label'] if can else None,
                target_set=candidate_target_set(can), added_distractor=can['sentence'] if can else '',
                must_not_be_used_for_training=True)
            if pool.get('fallback_mode') == 'fixed-template':
                row.update(candidate_origin=origin,
                    gate_validated=can.get('gate_validated') if can else None,
                    construction_fallback_used=origin == 'fixed_template_fallback',
                    fallback_trigger=can.get('fallback_trigger') if can else None,
                    fallback_template_version=can.get('fallback_template_version') if can else None,
                    random_same_target_control_collapsed=same_target_control_collapsed)
            output.append(row)
    return output, audits


def paired_rows(sources, views, audits):
    audits = {s['source_id']: s for s in audits}
    output = []
    for row in views:
        if row['view'] != 'hard': continue
        audit = audits[row['source_id']]
        pair = dict(question_id=row['source_id'], source_idx=row['source_idx'],
            clean_question=row['clean_question'], distracted_question=row['question'],
            options=row['options'], gold_answer=row['gold_answer'], intended_target=row['intended_target'],
            added_distractor=row['added_distractor'], candidate_id=row['candidate_id'],
            selection_pool='harmful_flip_priority' if audit['selected_by_flip_priority'] else 'maximum_D_fallback',
            construction_harmful_flip=audit['selected_by_flip_priority'], must_not_be_used_for_training=True)
        if 'candidate_origin' in row:
            pair.update(construction_candidate_origin=row['candidate_origin'],
                construction_fallback_used=row['construction_fallback_used'],
                gate_validated=row['gate_validated'], fallback_trigger=row['fallback_trigger'],
                fallback_template_version=row['fallback_template_version'],
                random_same_target_control_collapsed=row['random_same_target_control_collapsed'])
        output.append(pair)
    return output


# ------------------------------------------------------------------ vLLM backend for generation and the target check

class VllmRetryBackend:
    def __init__(self, chat_date="2026-09-12"):
        self.chat_date = chat_date
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        if importlib.metadata.version('vllm') != '0.24.0':
            raise RuntimeError('Unexpected vLLM version')
        self.params = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION,
                                                       trust_remote_code=True)
        self.llm = LLM(model=DEFAULT_MODEL, revision=DEFAULT_REVISION,
                       tokenizer_revision=DEFAULT_REVISION, trust_remote_code=True,
                       tensor_parallel_size=1, max_model_len=8192,
                       gpu_memory_utilization=.85, seed=42, max_num_seqs=8)
        self.extract = extract_final
        self.calls = dict(generate=0, judge=0)

    def render(self, prompt):
        rendered = self.tokenizer.apply_chat_template([dict(role='user',content=prompt)],
            tokenize=False,add_generation_prompt=True,reasoning_effort='low')
        # Freeze the automatic Harmony date across restarts.
        import re
        return re.sub(r'Current date: \d{4}-\d{2}-\d{2}', 'Current date: ' + self.chat_date, rendered)

    def generate(self, prompt, n, seed):
        self.calls['generate'] += 1
        rendered = self.render(prompt)
        output = self.llm.generate([rendered], self.params(n=n,temperature=.7,top_p=.95,
            seed=seed,max_tokens=512,skip_special_tokens=False),use_tqdm=False)[0].outputs
        return [dict(sentence=self.extract(r.text),raw_generation=r.text,
                     generated_token_ids=list(r.token_ids),finish_reason=r.finish_reason,
                     rendered_prompt_sha256=sha(rendered)) for r in output]

    def judge(self, prompt):
        self.calls['judge'] += 1
        rendered = self.render(prompt)
        r = self.llm.generate([rendered],self.params(n=1,temperature=0,seed=42,
            max_tokens=512,skip_special_tokens=False),use_tqdm=False)[0].outputs[0]
        parsed,error = parse_judgment(r.text)
        return dict(parsed=parsed,error=error,raw=r.text,tokens=list(r.token_ids),
                    finish_reason=r.finish_reason,rendered_prompt_sha256=sha(rendered))


# ------------------------------------------------------------------ Pipeline


def sources_from(rows, split):
    result = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not {'question', 'options', 'answer_idx'} <= row.keys():
            raise ValueError(f'Source row {i} needs question, options, and answer_idx')
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError(f'Source row {i} needs a nonempty question')
        options = row['options']
        if (not isinstance(options, dict) or set(options) != set('ABCD')
                or any(not isinstance(v, str) or not v.strip() for v in options.values())
                or row['answer_idx'] not in ('A', 'B', 'C', 'D')):
            raise ValueError(f'Source row {i} needs nonempty A-D option strings and a single A-D answer_idx')
        idx = int(row.get('idx', i))
        sid = (f'test-{idx:05d}' if split == 'test' else str(row.get('source_id', f'{split}-{idx:05d}')))
        result.append(dict(row, idx=idx, source_id=sid))
    if not result: raise ValueError('Empty source data')
    if len({r['source_id'] for r in result}) != len(result): raise ValueError('Duplicate source IDs')
    if len({r['idx'] for r in result}) != len(result): raise ValueError('Duplicate source indices')
    if len({sha([r['question'], r['options'], r['answer_idx']]) for r in result}) != len(result):
        raise ValueError('Duplicate source questions')
    return result


def settings(kind, split, retries=2, fallback_mode='none'):
    if kind not in ('bystander', 'nonliteral'): raise ValueError('Unknown distractor type')
    if fallback_mode not in ('none', 'fixed-template'): raise ValueError('Unknown fallback mode')
    prefix = 'nl' if kind == 'nonliteral' else 'bs'
    dataset_version = 'v2' if fallback_mode == 'fixed-template' else 'v1'
    return dict(kind=kind, split=split, policy=asdict(RetryPolicy(extra_rounds=retries, namespace=prefix+'-cue-retry-v2')),
        generation_template=NONLITERAL_PROMPT if kind == 'nonliteral' else prompt_gen_beta_confounder,
        judge_template=JUDGE_PROMPT.replace('bystander',kind).replace('Bystander',kind.title()),
        model=DEFAULT_MODEL, revision=DEFAULT_REVISION, chat_date='2026-09-12',
        namespace=f'{prefix}-{split}-{dataset_version}', random_salt=prefix+'-random-valid-v1',
        same_target_salt=prefix+'-random-same-target-v1', fallback_mode=fallback_mode,
        fallback_scope='source_without_gate_valid_candidates')


def generate_pool(sources, config, directory, backend):
    directory = Path(directory)
    # Keep the original source order and target order. Only rejected slots retry.
    results = []
    policy = RetryPolicy(**config['policy'])
    for i, source in enumerate(sources):
        for target in 'ABCD':
            if target == source['answer_idx']: continue
            results.append(run_target(source, target, config['generation_template'], config['judge_template'],
                policy, directory/'candidate_checkpoints', backend.generate, backend.judge,
                target_gate_result, dict(model=config['model'], revision=config['revision'], chat_date=config['chat_date'])))
        print(f"Candidates: {i+1}/{len(sources)}", flush=True)
    pool = build_scoring_pool(sources, results, insert_before_final_sentence, sample_by_hash,
                             condition=config['kind'], random_salt=config['random_salt'],
                             fallback_mode=config['fallback_mode'])
    jsonl(directory/'target_results.jsonl', results)
    atomic(directory/'pool.json', pool)
    jsonl(directory/'score_requests.jsonl', pool['score_requests'])
    return pool


def release_row(pair):
    """One row of the released MedQA-TD format."""
    return dict(id=pair['question_id'], question=pair['distracted_question'], options=pair['options'],
                answer=pair['gold_answer'], distractor=pair['added_distractor'], target=pair['intended_target'],
                clean_question=pair['clean_question'], fixed_template=bool(pair.get('construction_fallback_used', False)))


def finish(sources, config, state, output):
    pool = read(state/'pool.json')
    views, audit = select_dataset(sources, pool, read_jsonl(state/'scores.jsonl'),
        namespace=config['namespace'], random_salt=config['same_target_salt'])
    pairs = paired_rows(sources, views, audit)
    files = []
    jsonl(state/'selection_audit.jsonl',audit)
    selection=[{k:v for k,v in row.items() if not k.endswith('_hash')} for row in audit]
    full_coverage = config['fallback_mode'] == 'fixed-template'
    release = [release_row(p) for p in pairs]
    for name, rows in [(f"medqa-td-{config['kind']}.jsonl", release), ('pairs.jsonl', pairs), ('views.jsonl', views),
                       ('selection.jsonl', selection)]:
        jsonl(output/name,rows); files.append(output/name)
    summary = dict(type=config['kind'], sources=len(sources), included=len(pairs),
        excluded=len(sources)-len(pairs), extra_retry_rounds=config['policy']['extra_rounds'],
        generated_candidates=sum(r['generated_n'] for r in read_jsonl(state/'target_results.jsonl')),
        accepted_candidates=sum(c.get('candidate_origin') == 'generated_gate_valid' for c in pool['valid_candidates']))
    if full_coverage:
        summary.update(selection_eligible_candidates=len(pool['valid_candidates']),
            generated_source_coverage_n=pool['generated_source_coverage_n'],
            generated_source_coverage_rate=pool['generated_source_coverage_n']/len(sources),
            fallback_mode=config['fallback_mode'], fallback_sources=pool['fallback_source_n'],
            fallback_candidates=pool['fallback_candidate_n'], source_coverage_n=pool['source_coverage_n'],
            source_coverage_rate=pool['source_coverage_n']/len(sources),
            full_coverage=len(pairs) == len(sources))
    atomic(output/'summary.json', summary)
    files.append(output/'summary.json')
    if not pairs: raise ValueError('No eligible sources; selection audit saved')
    if full_coverage and len(pairs) != len(sources):
        raise ValueError('Fixed-template fallback did not achieve full source coverage')
    return files


def run(args):
    input_path = Path(args.input) if args.input else None
    split = args.split or ('internal' if input_path else 'test')
    source = None if input_path else medqa_source(split)
    config = settings(args.type,split,args.retry_rounds,getattr(args,'fallback_mode','none'))
    binding = dict(input=sha(input_path) if input_path else source, config=config)
    with Run(args.output, binding, args.resume) as run:
        state, output = run.state, run.output
        def prepare_sources():
            extra = []
            if input_path:
                rows = read_jsonl(input_path)
            else:
                rows, metadata = load_medqa(split, construction=True)
                atomic(state/'medqa_metadata.json', metadata)
                extra.append(state/'medqa_metadata.json')
            atomic(state/'sources.json', sources_from(rows, split))
            atomic(state/'config.json', config)
            return [state/'sources.json', state/'config.json'] + extra
        run.stage('prepare_sources', prepare_sources)
        sources = read(state/'sources.json')
        def candidate_stage():
            subprocess.run([getattr(args,'generation_python',sys.executable),'-m','medqa_td.build','--worker',str(state)],env=child_environment(),check=True)
            return [state/'target_results.jsonl',state/'pool.json',state/'score_requests.jsonl']
        run.stage('candidates',candidate_stage)
        def scoring_stage():
            env=child_environment(CUBLAS_WORKSPACE_CONFIG=':4096:8')
            subprocess.run([getattr(args,'scoring_python',sys.executable),'-m','medqa_td.build','--score',
                '--requests',str(state/'score_requests.jsonl'),'--output',str(state/'scores.jsonl'),
                '--manifest',str(state/'score_manifest.json')],env=env,check=True)
            validate_scores(read(state/'pool.json')['score_requests'],read_jsonl(state/'scores.jsonl'))
            return [state/'scores.jsonl',state/'score_manifest.json']
        run.stage('scoring',scoring_stage)
        def repeat_stage():
            # Independent process, same first 16 requests and exact A-D logprobs.
            requests=read_jsonl(state/'score_requests.jsonl')[:16]
            jsonl(state/'repeat_requests.jsonl',requests)
            env=child_environment(CUBLAS_WORKSPACE_CONFIG=':4096:8')
            subprocess.run([getattr(args,'scoring_python',sys.executable),'-m','medqa_td.build','--score',
                '--requests',str(state/'repeat_requests.jsonl'),'--output',str(state/'repeat_scores.jsonl'),
                '--manifest',str(state/'repeat_manifest.json')],env=env,check=True)
            repeated=validate_scores(requests,read_jsonl(state/'repeat_scores.jsonl'))
            baseline={r['request_id']:r for r in read_jsonl(state/'scores.jsonl')}
            if any(r['option_logprobs']!=baseline[k]['option_logprobs'] for k,r in repeated.items()):
                raise ValueError('Independent scoring repeat did not match')
            return [state/'repeat_requests.jsonl',state/'repeat_scores.jsonl',state/'repeat_manifest.json']
        run.stage('repeat_scoring',repeat_stage)
        run.stage('selection',lambda:finish(sources,config,state,output))
        print(f"Dataset complete: {read(output/'summary.json')['included']} questions -> {output}/medqa-td-{args.type}.jsonl")


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',help='Optional local Clean MedQA JSONL; otherwise download MedQA from Hugging Face')
    parser.add_argument('--type',required=True,choices=['bystander','nonliteral'])
    parser.add_argument('--split',choices=['train','dev','internal','test'],
                        help='HF split (default: test), or a local input label (default: internal; internal requires --input)')
    parser.add_argument('--retry-rounds',type=int,choices=[0,1,2],default=2)
    parser.add_argument('--fallback-mode',choices=['none','fixed-template'],default='none',
                        help='After the retry budget, fill only sources with zero gate-valid candidates using explicit fixed templates')
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--generation-python',default=sys.executable,help='Python with vLLM')
    parser.add_argument('--scoring-python',default=sys.executable,help='Python with native MXFP4 Transformers support')
    run(parser.parse_args())


if __name__ == '__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        state=Path(sys.argv[2]);config=read(state/'config.json')
        # Backend initializes lazily so a checkpoint-only replay needs no GPU.
        class LazyBackend:
            instance=None
            def call(self,name,*args):
                if self.instance is None: self.instance=VllmRetryBackend(config['chat_date'])
                return getattr(self.instance,name)(*args)
            def generate(self,*args): return self.call('generate',*args)
            def judge(self,*args): return self.call('judge',*args)
        generate_pool(read(state/'sources.json'),config,state,LazyBackend())
    elif len(sys.argv)>1 and sys.argv[1]=='--score':
        score_main(sys.argv[2:])
    else: main()
