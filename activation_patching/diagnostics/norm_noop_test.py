"""Test whether the paper's activation-scaling factor changes the model output.

Claim under test
----------------
The paper patches `mlp.down_proj.output` and multiplies it by a scaling factor c,
reporting that c>=2 drives the gender flip rate to 1.0. But on post-norm models
(Gemma-2, Gemma-3, MedGemma, OLMo-2) the MLP output passes through
`post_feedforward_layernorm` before it is added to the residual stream:

    pre-norm  (Llama, OLMo-1, Mistral, Phi, Qwen):
        residual += mlp_out                       -> scaling by c scales by c

    post-norm (Gemma-2/3, MedGemma, OLMo-2):
        residual += RMSNorm(mlp_out)              -> RMSNorm(c*x) == RMSNorm(x)

RMSNorm is scale-invariant, so on post-norm models c is annihilated before it
reaches the residual stream. Prediction: the logits are IDENTICAL across c.

The paper's Table 3 is consistent with this: Gemma-2 goes 0.83 -> 0.85 and
OLMo-2-32B goes 0.96 -> 0.96 as c sweeps 1 -> 20, while the two pre-norm models
both go to exactly 1.00.

This script tests it directly on real weights, with no vignette generation: patch
once per c, read the logits, compare. Four forward passes per configuration.

    python3 norm_noop_test.py -model_name allenai/OLMo-2-1124-7B-Instruct -layer 18

Expected output
---------------
  pre-norm  model, site=down_proj  -> logits move a lot,  p(Male) rises with c
  post-norm model, site=down_proj  -> logits IDENTICAL,   p(Male) flat  (the bug)
  post-norm model, site=post_norm  -> logits move a lot,  p(Male) rises (the fix)
"""

import argparse
import sys
from pathlib import Path

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_registry import (  # noqa: E402
    SUPPORTED_MODELS,
    derive_lm_head_depth,
    describe_architecture,
    get_patch_site,
    get_text_templates,
    is_post_norm,
    nnsight_logits,
)
from diagnostics.validate_model import build_prompts  # noqa: E402

SYSTEM = "You are a helpful clinical assistant."


def run_site(llm, site, prompts, patch_from, patch_to, layer, depth, answer_ids, factors):
    """Patch at `site` with each scaling factor; return logits + p(target) per c."""
    softmax = torch.nn.Softmax(dim=-1)
    clean_p, corrupt_p = prompts
    out = {}

    for c in factors:
        with torch.no_grad():
            with llm.generate(max_new_tokens=depth + 2) as tracer:
                with tracer.invoke(clean_p):
                    z_src = get_patch_site(llm, layer, site=site).output[:, patch_from, :]

                with tracer.invoke(corrupt_p):
                    mod = get_patch_site(llm, layer, site=site)
                    z = mod.output
                    z[:, patch_to, :] = z_src * c
                    mod.output = z

                    logits = nnsight_logits(llm, depth)
                    saved = logits[0][0].save()

        v = saved.float().cpu()
        out[c] = {
            "logits": v,
            "p_target": softmax(v)[answer_ids[0]].item(),
            "p_other": softmax(v)[answer_ids[1]].item(),
        }
    return out


def report(name, res, factors):
    base = res[factors[0]]["logits"]
    print(f"\n  --- patch site: {name} ---")
    print(f"  {'c':>4}  {'p(target)':>10}  {'p(other)':>10}  {'max|dlogit| vs c=1':>20}")
    for c in factors:
        d = (res[c]["logits"] - base).abs().max().item()
        print(f"  {c:>4}  {res[c]['p_target']:>10.4f}  {res[c]['p_other']:>10.4f}  {d:>20.6f}")

    deltas = [(res[c]["logits"] - base).abs().max().item() for c in factors]
    max_delta = max(deltas)
    p_spread = max(res[c]["p_target"] for c in factors) - min(res[c]["p_target"] for c in factors)

    # Verdict via the paper's own REWRITE SCORE:  (p* - p_1) / (1 - p_1)
    #
    # This normalizes the probability gain by the available headroom, so it reads correctly
    # whether the baseline p_1 is 0.32 (Gemma-2 @ L16) or 0.955 (OLMo-2-7B @ L6). A raw logit
    # delta is unreliable here (bf16 rounding alone gives |dlogit| ~0.1-0.25), and asking
    # whether |dlogit| grows with c penalizes success: a patch that saturates p(target) -> 1.0
    # stops moving the logits. The rewrite score has neither problem. Measured:
    #
    #   post-norm Gemma-2-9B,   down_proj: p 0.321 -> 0.294    rewrite -0.04  DEAD
    #   post-norm Gemma-2-9B,   post_norm: p 0.321 -> 0.9998   rewrite  1.00  ALIVE
    #   post-norm OLMo-2-7B,    down_proj: p 0.955 -> 0.958    rewrite  0.06  DEAD
    #   post-norm OLMo-2-7B,    post_norm: p 0.955 -> 0.998    rewrite  0.95  ALIVE
    #   post-norm OLMo-2-32B,   down_proj: p 0.950 -> 0.950    rewrite  0.00  DEAD
    #   pre-norm  OLMo-7B-0724, down_proj: p 0.068 -> 0.964    rewrite  0.96  ALIVE
    p1 = res[factors[0]]["p_target"]
    best = max(res[c]["p_target"] for c in factors)
    rewrite = (best - p1) / (1 - p1) if p1 < 1.0 else 0.0
    dead = rewrite < 0.10

    print(f"\n  => max|dlogit| = {max_delta:.4f}   (per c: "
          + ", ".join(f"{d:.3f}" for d in deltas) + ")")
    print(f"  => p(target): {p1:.4f} -> {best:.4f}   (spread {p_spread:.4f})")
    print(f"  => REWRITE SCORE (p* - p1)/(1 - p1) = {rewrite:+.4f}")
    if dead:
        print("  => scaling factor is A NO-OP: it does not move p(target) toward the target.")
    else:
        print(f"  => scaling factor is EFFECTIVE (recovers {100 * rewrite:.0f}% of the headroom).")
    return dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_name", choices=SUPPORTED_MODELS, required=True)
    ap.add_argument("-demographic_type", choices=["gender", "race"], default="gender")
    ap.add_argument("-condition", type=str, default="multiple sclerosis")
    ap.add_argument("-target", type=str, default="Male")
    ap.add_argument("-source", type=str, default="")
    ap.add_argument("-layer", type=int, required=True, help="peak layer from the heatmap")
    ap.add_argument("-factors", type=int, nargs="+", default=[1, 2, 5, 20])
    ap.add_argument("-load_in_4bit", action="store_true", default=True)
    ap.add_argument("-bf16", dest="load_in_4bit", action="store_false")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LanguageModel(args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
                        load_in_4bit=args.load_in_4bit)  # torch_dtype: transformers 4.49 compatibility

    arch = describe_architecture(llm, args.model_name)
    print(f"\n=== {args.model_name} ===")
    for k, v in arch.items():
        print(f"  {k:26} {v}")

    templates = get_text_templates(args.model_name)
    clean, corrupted, target_word, answers = build_prompts(
        tokenizer, templates, args.demographic_type, args.condition, args.target, args.source)

    clean_ids = llm.tokenizer(clean, return_tensors="pt")["input_ids"][0]
    corrupt_ids = llm.tokenizer(corrupted, return_tensors="pt")["input_ids"][0]
    answer_ids = [llm.tokenizer(a, add_special_tokens=False)["input_ids"][0] for a in answers]

    tgt_ids = llm.tokenizer(" " + target_word, return_tensors="pt")["input_ids"][0]
    patch_from = torch.argwhere(clean_ids == tgt_ids[-1])[0][0].item()

    cond_ids = llm.tokenizer(" " + args.condition, return_tensors="pt")["input_ids"][0]
    src_ix = -2 if args.condition in ("sarcoidosis", "rheumatoid arthritis") else -1
    patch_to = torch.argwhere(corrupt_ids == cond_ids[src_ix])[0][0].item()

    diff = len(clean_ids) - len(corrupt_ids)
    if diff <= 0:
        patch_from -= diff  # clean is shorter -> left-padded
    else:
        patch_to += diff

    depth = derive_lm_head_depth(llm, corrupted, answer_ids)

    print(f"\n  patch_from  {patch_from} ({llm.tokenizer.decode([clean_ids[patch_from + min(diff,0)]])!r})")
    print(f"  patch_to    {patch_to} -> writing into the condition token")
    print(f"  readout     depth={depth}, answers={answers} ids={answer_ids}")
    print(f"  layer       {args.layer}")

    prompts = (clean, corrupted)
    sites = ["down_proj"] + (["post_norm"] if is_post_norm(llm) else [])

    verdict = {}
    for site in sites:
        res = run_site(llm, site, prompts, patch_from, patch_to, args.layer,
                       depth, answer_ids, args.factors)
        verdict[site] = report(site, res, args.factors)

    print("\n" + "=" * 66)
    if is_post_norm(llm):
        print("POST-NORM MODEL. Expected: down_proj dead, post_norm effective.")
        ok = verdict.get("down_proj") is True and verdict.get("post_norm") is False
        print("RESULT: " + ("CONFIRMED -- the paper's scaling factor is a no-op here, "
                            "and patching post_feedforward_layernorm restores it."
                            if ok else
                            "NOT as predicted. down_proj dead=%s, post_norm dead=%s"
                            % (verdict.get("down_proj"), verdict.get("post_norm"))))
    else:
        print("PRE-NORM MODEL. Expected: down_proj effective (the paper's setting).")
        print("RESULT: " + ("as expected -- scaling works."
                            if verdict.get("down_proj") is False else
                            "UNEXPECTED -- scaling appears dead on a pre-norm model. "
                            "Something else is wrong; do not trust the patching runs."))


if __name__ == "__main__":
    main()
