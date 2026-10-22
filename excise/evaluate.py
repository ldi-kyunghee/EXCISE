"""EXCISE: Locating and Excising Distracting Sentences in Medical Question Answering.

Evaluate a model both without and with EXCISE on a JSONL file of multiple-choice questions.

Default input: four-option questions, with the final sentence treated as the question stem.
Each JSONL row needs ``id``, ``question``, ``options`` (A-D), and ``answer`` (gold letter).

    python -m excise.evaluate --model llama31_8b_inst --data data.jsonl --out runs/excise [--ablation]
    python -m excise.evaluate --model llama31_8b_inst --data data.jsonl --out runs/excise_cot --generate

Writes predictions.jsonl, summary.json, and resumable measured.jsonl + vectors.npz. The gate's basis (mean and first
principal component) is estimated on all narrative sentences of the input file, without labels. summary.json reports
accuracy without and with EXCISE, and with --ablation also EXCISE w/o IA and EXCISE w/o DG.
--generate: the answers come from chain-of-thought generation instead of the first-token readout (generations.jsonl).
        The sentence that EXCISE blocks in its first-token decision stays blocked during the
        generation (CoT + EXCISE); questions where it does not intervene keep the plain generation (CoT).

The method in the order of Section 5 of the paper:

1. Readout: the answer is read from the logits of the option letters at the answer position (first-token scoring).
2. Blocking a sentence: log(gamma) is added to the pre-softmax attention scores from every later position to the
   sentence's tokens, in all layers and heads (gamma = 0 removes that attention).
3. Interventional Attribution (IA): a_i = m - (l^(i)_o1 - l^(i)_o2), the drop of the decision margin when s_i is
   blocked; the candidate is the sentence with the largest a_i.
4. Discordance Gate (DG): c_i = cos(P h_i, P r_i) between the sentence read alone (h_i) and the rest of the narrative
   read alone (r_i), after P removes the mean and the first principal component of the h_i of an unlabeled batch;
   d_i = -(c_i - mean of the other c_j) / sd of the other c_j. EXCISE blocks the candidate only if d_i > tau.

Running EXCISE on a set of questions takes two passes: ``measure_question`` per question (readout, n blocked
passes, 2n short passes for h_i and r_i), then ``fit_basis`` on all h_i of the set and ``decide`` per question.
"""
import argparse
import json
import math
import re
import statistics
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------- models
# Hyperparameters chosen once per model on the development set (Appendix C.3, Table 9).
# layer: hidden-state layer of the gate (index into output_hidden_states; 0 = embeddings); tau: gate threshold;
# gamma: blocking strength. cot_format / cot_cache: prompt format and KV cache of chain-of-thought generation in the paper
# (Llama-3-Meerkat-8B ignores the JSON answer format and is prompted in its model card's format).
PRESETS = {
    "llama31_8b_inst": dict(hf_id="meta-llama/Llama-3.1-8B-Instruct", revision="0e9e39f249a16976918f6564b8830bc894c89659",
                       layer=9, tau=1.0, gamma=0.0, enable_thinking=None, cot_format="json", cot_cache=False),
    "qwen3_8b": dict(hf_id="Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218",
                     layer=25, tau=1.0, gamma=0.0, enable_thinking=False, cot_format="json", cot_cache=False),
    "meerkat_8b": dict(hf_id="dmis-lab/llama-3-meerkat-8b-v1.0", revision="ba8405f1d873b99004b2d121548fcafd33217eb7",
                       layer=19, tau=0.5, gamma=0.0, enable_thinking=None, cot_format="meerkat", cot_cache=True),
}
MIN_SENTENCES = 3   # questions with fewer narrative sentences are left unchanged


def load_model(model, device="cuda:0", hf_id=None, **overrides):
    """Return (model, tokenizer, preset): bfloat16, eager attention (needed for blocking), batch size 1.

    `model` is a PRESETS key or a compatible Hugging Face model id. The listed model ids use the same settings
    and pinned revision as their keys; other ids use untuned layer / tau / gamma defaults (16, 1.0, 0) unless
    supplied in `overrides`. `hf_id` replaces a preset's checkpoint location (e.g. a local snapshot directory).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    preset_key = model if model in PRESETS else next((k for k, p in PRESETS.items() if p["hf_id"] == model), None)
    if preset_key is not None:
        preset = dict(PRESETS[preset_key])
        name, rev = (hf_id, None) if hf_id else (preset["hf_id"], preset["revision"])
    else:
        preset = dict(hf_id=model, revision=None, layer=16, tau=1.0, gamma=0.0, enable_thinking=None, cot_format="json", cot_cache=False)
        name, rev = model, None
    preset.update({k: v for k, v in overrides.items() if v is not None})
    tokenizer = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, attn_implementation="eager",
                                                 device_map={"": device}, low_cpu_mem_usage=True)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer, preset


# ------------------------------------------------------------------------------------------- prompt
LETTERS = ("A", "B", "C", "D")
SYSTEM_PROMPT = "You are a medical AI assistant used for decision-making in a clinical setting."
RESPONSE_INSTRUCTION = (
    "Answer the following medical multiple-choice question.\n\n"
    'Return one JSON object with exactly two fields in this order: "rationale" and "answer".\n'
    "The answer must be one of A, B, C, or D."
)
RS_PREFIX = '{\n  "rationale": "",\n  "answer": "'   # teacher-forced assistant prefix; the next token is the answer letter


def option_letters(options):
    """A-D, or A-E for the five-option BiasMedQA questions."""
    letters = tuple(sorted(options))
    if letters not in (LETTERS, LETTERS + ("E",)):
        raise ValueError("options must have exactly the keys A-D or A-E")
    return letters


def messages(question, options, extra=None):
    """Chat messages; `extra` is an optional paragraph placed before "Question:" (used by the ignore-instruction baseline)."""
    letters = option_letters(options)
    instruction = RESPONSE_INSTRUCTION if len(letters) == 4 else \
        RESPONSE_INSTRUCTION.replace("one of A, B, C, or D.", "one of A, B, C, D, or E.")
    opts = "\n".join(f"{c}) {str(options[c]).strip()}" for c in letters)
    mid = f"{extra}\n\n" if extra else ""
    user = f"{instruction}\n\n{mid}Question:\n{str(question).strip()}\n\nOptions:\n{opts}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_prompt(tokenizer, question, options, enable_thinking=None, extra=None):
    """Return (chat text, token ids ending with RS_PREFIX). The chat template inserts BOS where needed."""
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    text = tokenizer.apply_chat_template(messages(question, options, extra), tokenize=False,
                                         add_generation_prompt=True, **kwargs)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] + tokenizer(RS_PREFIX, add_special_tokens=False)["input_ids"]
    return text, ids


def label_token_ids(tokenizer, letters=LETTERS):
    """Token id of each answer letter right after RS_PREFIX (each must be a single token there)."""
    base = tokenizer(RS_PREFIX, add_special_tokens=False)["input_ids"]
    out = {}
    for letter in letters:
        ids = tokenizer(RS_PREFIX + letter, add_special_tokens=False)["input_ids"]
        if ids[: len(base)] != base or len(ids) != len(base) + 1:
            raise ValueError(f"{letter!r} is not a single token after RS_PREFIX for this tokenizer")
        out[letter] = ids[-1]
    return out


# ---------------------------------------------------------------------------------------- sentences
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def narrative_sentences(question, stem_rule="last"):
    """Sentences of the case narrative: split after . ! ? followed by whitespace and a capital letter, then drop the
    question sentence. stem_rule "last": the last sentence (all MedQA-based sets); "last_question": the last sentence
    ending in "?" (BiasMedQA, whose bias cue follows the question; falls back to "last" if there is none)."""
    parts = [s.strip() for s in _SPLIT.split(normalize(question)) if s.strip()]
    if stem_rule == "last_question":
        q = [i for i, p in enumerate(parts) if p.endswith("?")]
        if q:
            return parts[:q[-1]] + parts[q[-1] + 1:]
    elif stem_rule != "last":
        raise ValueError(stem_rule)
    return parts[:-1] if len(parts) > 1 else parts


def token_span(tokenizer, text, sentence):
    """Token positions of `sentence` in `text`; None if it is not found exactly once or would include position 0
    (such sentences are not scored)."""
    if not sentence or text.count(sentence) != 1:
        return None
    start = text.index(sentence)
    end = start + len(sentence)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    span = [i for i, (a, b) in enumerate(offsets) if a < b and a < end and b > start]
    return None if not span or 0 in span else span


# ----------------------------------------------------------------------------------------- blocking
def _edit_mask(mask, sources, bias):
    """Copy of the additive 4D mask (1, heads|1, q_len, k_len) with `bias` on (later query, source key) entries."""
    import torch

    if mask is None or mask.dim() != 4 or mask.shape[0] != 1:
        raise RuntimeError("blocking needs the explicit 4D eager attention mask of a single sequence")
    q_len, k_len = mask.shape[-2], mask.shape[-1]
    offset = k_len - q_len          # absolute position of the first query row (0 without a KV cache)
    src = sorted(set(sources))
    rows = [j for j in range(q_len) if offset + j > src[0] and offset + j not in src]
    if not rows:
        return mask
    r = torch.tensor(rows, device=mask.device)[:, None]
    s = torch.tensor(src, device=mask.device)[None, :]
    if not torch.all(mask[:, :, r, s] == 0):
        raise RuntimeError("blocked edges must be unmasked before blocking")
    out = mask.clone()
    out[:, :, r, s] = bias
    return out


@contextmanager
def block_sentence(model, sources, gamma):
    """While active, every forward pass of `model` blocks the token positions `sources` with strength gamma."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if model.config._attn_implementation != "eager":
        raise RuntimeError('load the model with attn_implementation="eager"')
    if not sources or 0 in sources:
        raise ValueError("sources must be non-empty and must not include position 0")
    if gamma == 1.0:
        yield
        return
    bias = float("-inf") if gamma == 0.0 else math.log(gamma)

    def hook(module, args, kwargs):
        if "attention_mask" not in kwargs:
            raise RuntimeError("attention module did not receive attention_mask as a keyword argument")
        kwargs["attention_mask"] = _edit_mask(kwargs["attention_mask"], sources, bias)
        return args, kwargs

    handles = [layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True) for layer in model.model.layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ------------------------------------------------------------------------------------------ readout
def option_logits(model, ids, label_ids, block=None, gamma=0.0):
    """Logits of the option letters at the answer position, optionally with the positions `block` blocked."""
    import torch

    x = torch.tensor([list(ids)], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode():
        if block is None:
            logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
        else:
            with block_sentence(model, block, gamma):
                logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
    last = logits[0, -1].float().cpu()
    return {k: float(last[i]) for k, i in label_ids.items()}


def readout(model, tokenizer, question, options, preset):
    """Answer without intervention: (letter, logits of the option letters)."""
    _, ids = build_prompt(tokenizer, normalize(question), options, preset.get("enable_thinking"))
    z = option_logits(model, ids, label_token_ids(tokenizer, option_letters(options)))
    return max(z, key=z.get), z


def mean_hidden(model, tokenizer, text, layer):
    """Mean over tokens of the hidden states at `layer` for `text` read alone (no prompt), stored in float16."""
    import torch

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode():
        hs = model(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states
    return hs[layer][0].float().mean(0).to(torch.float16).cpu().numpy()


# ------------------------------------------------------------------------- Interventional Attribution
def measure_question(model, tokenizer, question, options, preset, stem_rule="last"):
    """Readout, Interventional Attribution and the gate's vectors for one question.

    Per narrative sentence: the attribution score ``a`` (None if the sentence could not be located), the answer of
    its blocked pass, and the vectors ``h`` (sentence alone) and ``r`` (rest of the narrative alone).
    """
    gamma, layer = preset["gamma"], preset["layer"]
    label_ids = label_token_ids(tokenizer, option_letters(options))
    q = normalize(question)
    text, ids = build_prompt(tokenizer, q, options, preset.get("enable_thinking"))
    z0 = option_logits(model, ids, label_ids)
    o1, o2 = sorted(z0, key=z0.get, reverse=True)[:2]      # top two options, fixed before any intervention
    margin = z0[o1] - z0[o2]
    narr = narrative_sentences(q, stem_rule)
    sents = []
    for i, s in enumerate(narr):
        span = token_span(tokenizer, text, s)
        row = dict(sentence=s, a=None, answer=None)
        if span is not None:
            z1 = option_logits(model, ids, label_ids, block=span, gamma=gamma)
            row["a"] = round(margin - (z1[o1] - z1[o2]), 4)   # rounded to 4 decimals as in the paper's runs
            row["answer"] = max(z1, key=z1.get)
        rest = " ".join(narr[:i] + narr[i + 1:]).strip()
        row["h"] = mean_hidden(model, tokenizer, s, layer)
        row["r"] = mean_hidden(model, tokenizer, rest if rest else ".", layer)
        sents.append(row)
    return dict(logits=z0, answer=o1, margin=margin, sentences=sents)


# ---------------------------------------------------------------------------------- Discordance Gate
def fit_basis(h_vectors, iters=30, seed=0):
    """Mean and first principal component (power iteration, as in the paper's runs) of an unlabeled batch of h_i."""
    h = np.asarray(h_vectors, dtype=np.float64)
    mu = h.mean(0)
    xc = h - mu
    u = np.random.default_rng(seed).normal(size=xc.shape[1])
    u /= np.linalg.norm(u)
    for _ in range(iters):
        u = xc.T @ (xc @ u)
        u /= np.linalg.norm(u)
    return mu, u


def _project(x, mu, u):
    x = np.asarray(x, dtype=np.float64) - mu
    x = x - np.outer(x @ u, u)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def similarities(measured, basis):
    """c_i = cos(P h_i, P r_i) for every narrative sentence of one measured question."""
    if not measured["sentences"]:
        return []
    mu, u = basis
    h = _project([s["h"] for s in measured["sentences"]], mu, u)
    r = _project([s["r"] for s in measured["sentences"]], mu, u)
    return [float(a @ b) for a, b in zip(h, r)]


def discordance(sims, i):
    """d_i = -(c_i - mean of the other c_j) / population sd of the other c_j; None with fewer than two others."""
    others = [x for j, x in enumerate(sims) if j != i and x is not None]
    if len(others) < 2 or sims[i] is None:
        return None
    m, sd = statistics.mean(others), statistics.pstdev(others)
    return -((sims[i] - m) / sd) if sd > 1e-9 else 0.0


# ------------------------------------------------------------------------------------------ decision
def decide(measured, basis, tau, min_sentences=MIN_SENTENCES, variant="excise"):
    """Final answer for one measured question; returns (answer, index of the blocked sentence or None).

    variant: "excise" (IA selects, DG gates), "no_dg" (block the top-attributed sentence in every eligible question),
    "no_ia" (block the most discordant sentence if it passes the gate). Ties go to the earlier sentence.
    """
    sents = measured["sentences"]
    valid = [i for i, s in enumerate(sents) if s["a"] is not None]
    if len(valid) < min_sentences:
        return measured["answer"], None
    sims = similarities(measured, basis)
    d = [discordance(sims, i) for i in range(len(sents))]
    if variant == "no_ia":
        cand = [i for i in valid if d[i] is not None]
        if not cand:
            return measured["answer"], None
        gi = max(cand, key=lambda i: (d[i], -i))
        return (sents[gi]["answer"], gi) if d[gi] > tau else (measured["answer"], None)
    ti = max(valid, key=lambda i: sents[i]["a"])
    if variant == "no_dg":
        return sents[ti]["answer"], ti
    if variant != "excise":
        raise ValueError(variant)
    if d[ti] is not None and d[ti] > tau:
        return sents[ti]["answer"], ti
    return measured["answer"], None


# ------------------------------------------------------------------------------- chain-of-thought generation
# Generation mode of the paper (Table 2, CoT rows): the readout prompt without the answer prefix, greedy decoding of at
# most 1,024 new tokens; CoT + EXCISE keeps the sentence that EXCISE blocks in its first-token decision blocked during
# the whole generation.
MAX_NEW_TOKENS = 1024
STOP_TOKENS = ("<|end_of_text|>", "<|eom_id|>", "<|eot_id|>", "<|im_end|>")
MEERKAT_SYSTEM = (   # system message of the Llama-3-Meerkat-8B model card
    "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting "
    "by summarizing the available information. Output a single option from the given options as the final answer. You "
    "are strongly required to follow the specified output format; conclude your response with the phrase \"the answer "
    "is ([option_id]) [answer_string]\".\n\n"
)
_FINAL_MARKERS = ("## Final Response", "<unused95>")
_FENCED_JSON = re.compile(r"\A```json\s*(\{.*\})\s*```\Z", flags=re.IGNORECASE | re.DOTALL)


def cot_prompt(tokenizer, question, options, preset):
    """(chat text, token ids) of the generation prompt."""
    q = normalize(question)
    if preset.get("cot_format") == "meerkat":
        user = f"{q} " + " ".join(f"({c}) {options[c]}" for c in option_letters(options))
        msgs = [{"role": "system", "content": MEERKAT_SYSTEM}, {"role": "user", "content": user}]
    else:
        msgs = messages(q, options)
    kwargs = {} if preset.get("enable_thinking") is None else {"enable_thinking": preset["enable_thinking"]}
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
    return text, tokenizer(text, add_special_tokens=False)["input_ids"]


def parse_answer(text, letters, cot_format="json"):
    """Answer letter of a generated response, or None. "json": the "answer" field of the JSON object (optionally in one
    ```json fence or after a native final-response marker; otherwise a JSON-shaped text ending in its only "answer"
    field). "meerkat": the last "the answer is (X)" (the opening parenthesis and a colon are optional)."""
    L = "".join(letters)
    if cot_format == "meerkat":
        m = re.findall(rf"he answer is:?\s*\(?([{L}])\)", text)
        return m[-1] if m else None
    candidate = str(text).strip()
    located = [(candidate.rfind(m), m) for m in _FINAL_MARKERS if candidate.rfind(m) >= 0]
    if located:
        idx, marker = max(located)
        candidate = candidate[idx + len(marker):].strip()
    fenced = _FENCED_JSON.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group(1).strip()
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        answer = str(payload.get("answer", "")).strip().upper()
        if list(payload) == ["rationale", "answer"] and isinstance(payload.get("rationale"), str) and answer in letters:
            return answer
    field = re.compile(rf'["\']answer["\']\s*:\s*["\']([{L}])["\']', flags=re.IGNORECASE)
    final = re.compile(rf'["\']answer["\']\s*:\s*["\']([{L}])["\']\s*\}}?\s*\Z', flags=re.IGNORECASE)
    matches = field.findall(candidate) if candidate.startswith("{") and final.search(candidate) else []
    return matches[0].upper() if len(matches) == 1 else None


def generate_cot(model, tokenizer, question, options, preset, block_sentence_text=None, use_cache=None):
    """Greedy chain-of-thought generation; with `block_sentence_text`, that narrative sentence stays blocked throughout.

    Without the KV cache every step recomputes the full sequence (the paper's setting for Llama-3.1-8B-Instruct and
    Qwen3-8B on MedQA and MedQA-TD); returns dict(answer, text, n_generated, truncated)."""
    import torch

    use_cache = preset.get("cot_cache", False) if use_cache is None else use_cache
    text, ids = cot_prompt(tokenizer, question, options, preset)
    span = None
    if block_sentence_text is not None:
        span = token_span(tokenizer, text, block_sentence_text)
        if span is None:
            raise RuntimeError("the blocked sentence does not occur exactly once in the generation prompt")
    vocab = tokenizer.get_vocab()
    stops = ({tokenizer.eos_token_id} | {vocab[t] for t in STOP_TOKENS if t in vocab}) - {None}
    dev = next(model.parameters()).device
    out = []
    ctx = block_sentence(model, span, preset["gamma"]) if span else nullcontext()
    with ctx, torch.inference_mode():
        if not use_cache:
            for _ in range(MAX_NEW_TOKENS):
                x = torch.tensor([ids + out], dtype=torch.long, device=dev)
                logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[:, -1, :].float().cpu()[0]
                out.append(int(torch.argmax(logits)))
                if out[-1] in stops:
                    break
        else:
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=True)
            for _ in range(MAX_NEW_TOKENS):
                out.append(int(torch.argmax(o.logits[0, -1].float())))
                if out[-1] in stops or len(out) == MAX_NEW_TOKENS:
                    break
                mask = torch.ones((1, len(ids) + len(out)), dtype=torch.long, device=dev)
                o = model(input_ids=torch.tensor([[out[-1]]], device=dev), attention_mask=mask,
                          past_key_values=o.past_key_values, use_cache=True)
    gen = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    stopped = bool(out and out[-1] in stops)
    return dict(answer=parse_answer(gen, option_letters(options), preset.get("cot_format", "json")), text=gen,
                n_generated=len(out), truncated=len(out) >= MAX_NEW_TOKENS and not stopped)


# --------------------------------------------------------------------------------------- evaluation

def load_rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def load_measured(out):
    """Measured questions of a run directory (with the gate vectors h, r attached), keyed by id."""
    out = Path(out)
    mpath, vpath = out / "measured.jsonl", out / "vectors.npz"
    if not (mpath.exists() and vpath.exists()):
        return {}
    vec = np.load(vpath)
    done = {}
    for line in open(mpath):
        m = json.loads(line)
        if m["sentences"] and f"{m['id']}|0|h" not in vec:
            continue                       # measured after the last vector checkpoint; measured again
        for k, s in enumerate(m["sentences"]):
            s["h"], s["r"] = vec[f"{m['id']}|{k}|h"], vec[f"{m['id']}|{k}|r"]
        done[m["id"]] = m
    return done


def _slim(m):
    return dict(m, sentences=[{x: y for x, y in s.items() if x not in ("h", "r")} for s in m["sentences"]])


def measure_file(model, tokenizer, preset, rows, out, stem_rule=None, save_every=100):
    """Measure rows, inferring the A-D/A-E question-stem rule unless explicitly supplied."""
    done = load_measured(out)
    vectors = {f"{i}|{k}|{x}": s[x] for i, m in done.items() for k, s in enumerate(m["sentences"]) for x in ("h", "r")}
    todo = [r for r in rows if r["id"] not in done]
    with open(out / "measured.jsonl", "w") as fh:          # rewritten so that it matches vectors.npz
        fh.writelines(json.dumps(_slim(m)) + "\n" for m in done.values())
        for n, r in enumerate(todo, 1):
            rule = stem_rule or ("last_question" if len(r["options"]) == 5 else "last")
            m = measure_question(model, tokenizer, r["question"], r["options"], preset, rule)
            m["id"], m["gold"] = r["id"], r["answer"]
            for k, s in enumerate(m["sentences"]):
                vectors[f"{r['id']}|{k}|h"], vectors[f"{r['id']}|{k}|r"] = s["h"], s["r"]
            fh.write(json.dumps(_slim(m)) + "\n")
            done[r["id"]] = m
            if n % save_every == 0 or n == len(todo):
                fh.flush()
                np.savez(out / "vectors.npz", **vectors)
                print(f"measured {len(done)}/{len(rows)}", flush=True)
    return [done[r["id"]] for r in rows]


def generate_file(model, tok, preset, rows, preds, out, use_cache=None, stem_rule=None):
    """Chain-of-thought answers (resumable via generations.jsonl). For EXCISE, the blocked sentence of the first-token
    decision stays blocked; elsewhere the plain generation is used."""
    path = out / "generations.jsonl"
    done = {(g["id"], g["blocked"]): g for g in map(json.loads, open(path))} if path.exists() else {}
    with open(path, "a") as fh:
        def gen(r, k):
            if (r["id"], k) not in done:
                rule = stem_rule or ("last_question" if len(r["options"]) == 5 else "last")
                sentence = None if k is None else narrative_sentences(normalize(r["question"]), rule)[k]
                g = dict(id=r["id"], blocked=k, **generate_cot(model, tok, r["question"], r["options"], preset, sentence, use_cache))
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
                fh.flush()
                done[(r["id"], k)] = g
            return done[(r["id"], k)]["answer"]

        new = []
        for n, (r, p) in enumerate(zip(rows, preds), 1):
            q = dict(id=r["id"], gold=r["answer"], base=gen(r, None))
            if "excise" in p:
                q["excise"], q["excise_blocked"] = (q["base"], None) if p["excise_blocked"] is None else (gen(r, p["excise_blocked"]), p["excise_blocked"])
            new.append(q)
            if n % 25 == 0:
                print(f"generated {n}/{len(rows)}", flush=True)
    return new


def score(measured, preset, variants):
    basis = fit_basis([s["h"] for m in measured for s in m["sentences"]])
    preds = []
    for m in measured:
        p = dict(id=m["id"], gold=m["gold"], base=m["answer"])
        for v in variants:
            p[v], p[f"{v}_blocked"] = decide(m, basis, preset["tau"], MIN_SENTENCES, v)
        preds.append(p)
    return preds


def summarize(preds, keys):
    acc = lambda k: round(100 * float(np.mean([p[k] == p["gold"] for p in preds])), 2)
    s = {"n": len(preds), "accuracy": {k: acc(k) for k in keys}}
    blocked = [k for k in keys if f"{k}_blocked" in preds[0]] if preds else []
    if blocked:
        s["intervened_pct"] = {k: round(100 * float(np.mean([p[f"{k}_blocked"] is not None for p in preds])), 1) for k in blocked}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a listed model key or a compatible Hugging Face model id")
    ap.add_argument("--data", required=True, help="path to the evaluation JSONL file")
    ap.add_argument("--out", required=True, help="directory for results and checkpoints; created if missing")
    ap.add_argument("--ablation", action="store_true", help="also report w/o IA and w/o DG accuracies using logits at the JSON answer-letter position")
    ap.add_argument("--generate", action="store_true", help="generate a rationale and final answer, then extract the answer instead of directly reading answer-letter logits")
    a = ap.parse_args()
    model, tok, preset = load_model(a.model)
    rows = load_rows(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = ("excise", "no_ia", "no_dg") if a.ablation else ("excise",)
    preds = score(measure_file(model, tok, preset, rows, out), preset, variants)
    keys = ["base", *variants]
    if a.generate:
        preds = generate_file(model, tok, preset, rows, preds, out)
        keys = [k for k in ("base", "excise") if k in preds[0]]
    with open(out / "predictions.jsonl", "w") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in preds)
    summary = dict(summarize(preds, keys), method="excise", model=a.model, readout="cot" if a.generate else "first_token",
                   **{k: preset[k] for k in ("layer", "tau", "gamma")})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
