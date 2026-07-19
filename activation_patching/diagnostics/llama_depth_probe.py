"""Locate the gender token in a model's generation to set the rewrite-score readout depth.

The rewrite score is read at a fixed generated-token index (the readout depth). This probe
generates greedily from the exact destination prompt get_patching_scores uses and reports
the index at which the model emits the gender word, which is the correct depth; reading at
any other index scores a non-demographic token. Under greedy decoding the four paper models
emit the gender word at hop 2.

Run from activation_patching:  python3 -u diagnostics/llama_depth_probe.py
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from nnsight import LanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_registry import get_text_templates, derive_lm_head_depth  # noqa: E402

SYSTEM = "You are a helpful clinical assistant."


def dest_prompt(tokenizer, model_name, condition):
    """The vignette-generation prompt get_patching_scores reads the gender token from."""
    tt = get_text_templates(model_name)
    role = tt["role"]
    text = tt["gender"].format(condition)
    messages = [{"role": role, "content": "{0}\n\n{1}".format(SYSTEM, text)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("-conditions", nargs="+", default=["multiple sclerosis", "sarcoidosis"])
    ap.add_argument("-target", default="male")
    ap.add_argument("-max_new_tokens", type=int, default=12)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LanguageModel(args.model_name, torch_dtype=torch.bfloat16, device_map="auto", load_in_4bit=True)

    pair = (" Male", " Female") if args.target == "male" else (" Female", " Male")
    answer_ids = [tokenizer(a, add_special_tokens=False)["input_ids"][0] for a in pair]
    gender_words = {"male", "female", "man", "woman"}

    for cond in args.conditions:
        prompt = dest_prompt(tokenizer, args.model_name, cond)
        # DETERMINISTIC greedy generation -- Llama's config otherwise samples (temp 0.6),
        # which would make the gender position (and thus any fixed readout depth) unstable.
        with torch.no_grad():
            with llm.generate(prompt, max_new_tokens=args.max_new_tokens, do_sample=False):
                out = llm.generator.output.save()
        seq = out[0] if out.ndim == 2 else out
        new_ids = [int(t) for t in seq[-args.max_new_tokens:]]
        decoded = [tokenizer.decode([t]) for t in new_ids]

        print(f"\n=================  {cond}  =================")
        print("  greedy generation, token by token (hop = readout depth):")
        gender_hop = None
        for hop, (tid, tok) in enumerate(zip(new_ids, decoded)):
            mark = ""
            if tok.strip().lower() in gender_words:
                mark = "   <=====  GENDER WORD"
                if gender_hop is None:
                    gender_hop = hop
            print(f"     hop {hop:2d}: {tok!r}{mark}")
        print(f"  --> gender word first appears at hop {gender_hop}")
        try:
            d = derive_lm_head_depth(llm, prompt, answer_ids)
            print(f"  --> derive_lm_head_depth() returns {d}")
        except Exception as e:
            print("  --> derive_lm_head_depth() FAILED:", str(e).splitlines()[0])
        print("  (get_patching_scores reads the rewrite score at this depth)")


if __name__ == "__main__":
    main()
