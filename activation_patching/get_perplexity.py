"""Fluency guard: perplexity of patched vignettes per scaling factor.

The paper reports this measurement (its Fig. 3b / Fig. 8) but does not release the code.
Fluent patching leaves perplexity on the unpatched baseline; a deliberately corrupted
patch gives a high reference (~15.5). The judge must be from a different model family than
the vignette's generator (the paper judges with Llama-3.1-8B and avoids scoring OLMo with
OLMo, per Panickssery et al.), so Llama vignettes are judged by OLMo and the others by
Llama.

    python3 get_perplexity.py -ia_csv <IA_*.csv> -judge meta-llama/Llama-3.1-8B-Instruct

Reads the `text` and `factor` columns, strips the prompt, and reports mean perplexity per
factor.
"""

import argparse
import glob
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Where the model's own turn begins, per chat template. We score ONLY the generated
# vignette -- including the prompt would swamp the signal with identical tokens.
ASSISTANT_MARKERS = [
    "<|assistant|>",           # OLMo-1 / OLMo-2
    "<start_of_turn>model",    # Gemma
    "<|start_header_id|>assistant<|end_header_id|>",  # Llama-3
    "[/INST]",                 # Mistral
]


def strip_prompt(text: str) -> str:
    for m in ASSISTANT_MARKERS:
        if m in text:
            text = text.split(m)[-1]
    # drop trailing special tokens
    text = re.sub(r"<\|endoftext\|>|<eos>|<\|eot_id\|>|</s>", "", text)
    return text.strip()


@torch.no_grad()
def perplexity(model, tok, text: str, max_len: int = 512) -> float:
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids
    if ids.shape[1] < 2:
        return float("nan")
    ids = ids.to(model.device)
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-ia_csv", required=True, nargs="+",
                    help="IA_*.csv path(s) or a glob; globs are expanded in Python, so a "
                         "quoted pattern also works.")
    ap.add_argument("-judge", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="judge LM. MUST be a different family from the model under test.")
    ap.add_argument("-max_per_group", type=int, default=100,
                    help="vignettes to score per factor (perplexity is stable well before 500)")
    args = ap.parse_args()

    paths = []
    for p in args.ia_csv:
        hits = sorted(glob.glob(p))
        paths.extend(hits if hits else ([p] if os.path.exists(p) else []))
    if not paths:
        raise SystemExit(f"No CSVs matched: {args.ia_csv}")
    print(f"Scoring {len(paths)} file(s):")
    for p in paths:
        print("   ", p)

    tok = AutoTokenizer.from_pretrained(args.judge)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge, torch_dtype=torch.bfloat16, device_map="auto")  # torch_dtype for transformers 4.49 (repro env)
    model.eval()

    for path in paths:
        df = pd.read_csv(path, sep="\t")
        under_test = path.split("_")[-1].replace(".csv", "")
        if under_test.split("-")[0].lower() in args.judge.lower():
            print(f"\n*** REFUSING {path}: judge {args.judge} is the same family as the model\n"
                  f"    under test ({under_test}). Self-scoring inflates the result.\n")
            continue

        print("\n" + "=" * 66)
        print(path.split("/")[-1])
        print("=" * 66)
        rows = []
        for f, grp in df.groupby("factor"):
            texts = [strip_prompt(t) for t in grp["text"].head(args.max_per_group)]
            texts = [t for t in texts if len(t) > 30]
            ppl = [perplexity(model, tok, t) for t in texts]
            ppl = [p for p in ppl if p == p]  # drop nan
            rows.append(dict(factor=f, n=len(ppl),
                             mean_ppl=sum(ppl) / len(ppl) if ppl else float("nan"),
                             median_ppl=sorted(ppl)[len(ppl) // 2] if ppl else float("nan")))

        r = pd.DataFrame(rows)
        base = r.iloc[0]["mean_ppl"]     # lowest factor/alpha = closest to unpatched
        r["vs_weakest"] = (r["mean_ppl"] / base).round(3)
        print(r.to_string(index=False))
        print()
        print("Interpretation: perplexity should stay FLAT as the factor/alpha rises. A rising")
        print("curve means the patch is corrupting the text, so any apparent bias reduction is")
        print("not a valid result. (The paper's broken-patch control sits at ppl 15.54.)")


if __name__ == "__main__":
    main()
