"""Pre-flight check for a model before a patching run.

Verifies and prints every assumption the patching scripts make, so a misconfiguration
shows up here rather than as a silently flat heatmap.

    python3 validate_model.py -model_name allenai/OLMo-7B-0724-Instruct-hf \
        -demographic_type gender -condition "multiple sclerosis" -target Male

Checks, in order:
  1. nnsight loads the model and the decoder layers are findable
  2. architecture: pre-norm vs post-norm MLP (decides the patch site)
  3. the source token (' Male') is where we think it is in the clean prompt
  4. the destination token (last subtoken of the condition) is where we think it is
  5. the readout position actually emits the demographic (derived, not assumed)
  6. the MLP output at the patch site has RMS >> rms_norm_eps
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
    get_layers,
    get_patch_site,
    get_text_templates,
)

SYSTEM = "You are a helpful clinical assistant."


def build_prompts(tokenizer, templates, demographic_type, condition, target, source):
    """Clean (source of activations) and corrupted (vignette) prompts.

    Mirrors get_patching_scores.py; keep the two in sync.
    """
    role = templates["role"]
    if demographic_type == "gender":
        target_word = "Male" if target.lower() == "male" else "Female"
        answers = (" Male", " Female") if target.lower() == "male" else (" Female", " Male")
    else:
        target_word = target[0].upper() + target[1:].lower()
        src_word = source[0].upper() + source[1:].lower()
        answers = (" " + target_word, " " + src_word)

    clean = tokenizer.apply_chat_template(
        [{"role": role, "content": f"The patient is {target_word}."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    corrupted = tokenizer.apply_chat_template(
        [{"role": role, "content": "{0}\n\n{1}".format(
            SYSTEM, templates[demographic_type].format(condition))}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return clean, corrupted, target_word, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_name", choices=SUPPORTED_MODELS, required=True)
    ap.add_argument("-demographic_type", choices=["gender", "race"], default="gender")
    ap.add_argument("-condition", type=str, default="multiple sclerosis")
    ap.add_argument("-target", type=str, default="Male")
    ap.add_argument("-source", type=str, default="")
    ap.add_argument("-layer", type=int, default=18, help="layer to sample activation RMS at")
    ap.add_argument("-load_in_4bit", action="store_true", default=True)
    ap.add_argument("-bf16", dest="load_in_4bit", action="store_false",
                    help="load in bf16 instead of 4-bit")
    args = ap.parse_args()

    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print(f"\n=== validating {args.model_name} ({args.demographic_type}) ===\n")

    # nnsight reaches into HF internals and the decoder-layer output type changed across
    # the transformers 4.5x series, so report the loaded versions. The paper used 4.49.0.
    from importlib.metadata import PackageNotFoundError, version as pkg_version  # noqa: E402
    import transformers  # noqa: E402

    def _ver(name):
        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return "?"

    print("[0/6] environment")
    print(f"    nnsight       {_ver('nnsight')}")
    print(f"    transformers  {transformers.__version__}")
    print(f"    torch         {torch.__version__}")
    print(f"    quantization  {'4-bit' if args.load_in_4bit else 'bf16'}")

    import huggingface_hub  # noqa: E402
    print(f"    hf cache      {huggingface_hub.constants.HF_HUB_CACHE}")

    if not transformers.__version__.startswith("4.49"):
        print("    NOTE: the paper used transformers 4.49.0; other versions may differ.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LanguageModel(
        args.model_name,
        torch_dtype=torch.bfloat16,  # torch_dtype: transformers 4.49 compatibility
        device_map="auto",
        load_in_4bit=args.load_in_4bit,
    )

    # ---- 1 & 2: architecture -------------------------------------------------
    print("[1/6] architecture")
    arch = describe_architecture(llm, args.model_name)
    for k, v in arch.items():
        print(f"    {k:26} {v}")
    if arch["post_norm_mlp"]:
        print("    NOTE: post-norm model. The MLP output is normalized before the residual\n"
              "          add, so scaling a patched mlp.down_proj.output by c does NOTHING.\n"
              "          Patch site auto-selected: post_feedforward_layernorm.")
    n_layers = arch["n_layers"]
    check("layer index in range", 0 <= args.layer < n_layers, f"layer={args.layer}, n_layers={n_layers}")

    templates = get_text_templates(args.model_name)
    clean, corrupted, target_word, answers = build_prompts(
        tokenizer, templates, args.demographic_type, args.condition, args.target, args.source)

    clean_ids = llm.tokenizer(clean, return_tensors="pt")["input_ids"][0]
    corrupt_ids = llm.tokenizer(corrupted, return_tensors="pt")["input_ids"][0]

    answer_ids = [llm.tokenizer(a, add_special_tokens=False)["input_ids"][0] for a in answers]
    print(f"\n    answer tokens: {answers} -> ids {answer_ids}")

    # ---- 3: source token position -------------------------------------------
    print("\n[2/6] source token (where we READ the activation from)")
    tgt_ids = llm.tokenizer(" " + target_word, return_tensors="pt")["input_ids"][0]
    hits = torch.argwhere(clean_ids == tgt_ids[-1])
    if len(hits) == 0:
        check("source token found in clean prompt", False,
              f"token {tgt_ids[-1].item()} ({target_word!r}) absent")
        patch_from = None
    else:
        patch_from = hits[0][0].item()
        decoded = llm.tokenizer.decode([clean_ids[patch_from]])
        check("source token position", decoded.strip().lower() == target_word.lower(),
              f"index {patch_from} decodes to {decoded!r} (want {target_word!r})")
        if len(hits) > 1:
            print(f"    WARNING: {len(hits)} occurrences; upstream takes the first. "
                  f"positions={[h[0].item() for h in hits]}")

    # ---- 4: destination token position --------------------------------------
    print("\n[3/6] destination token (where we WRITE the activation to)")
    cond_ids = llm.tokenizer(" " + args.condition, return_tensors="pt")["input_ids"][0]
    # upstream hardcodes source_ix=-2 for sarcoidosis / rheumatoid arthritis / prostate cancer
    for src_ix in (-1, -2):
        hits = torch.argwhere(corrupt_ids == cond_ids[src_ix])
        if len(hits):
            pos = hits[0][0].item()
            print(f"    source_ix={src_ix:>2} -> corrupted index {pos} "
                  f"decodes to {llm.tokenizer.decode([corrupt_ids[pos]])!r}")
    hits = torch.argwhere(corrupt_ids == cond_ids[-1])
    check("condition's last subtoken present in vignette prompt", len(hits) > 0,
          f"{args.condition!r} -> {[llm.tokenizer.decode([i]) for i in cond_ids]}")

    # ---- 5: padding offset ---------------------------------------------------
    diff = len(clean_ids) - len(corrupt_ids)
    print(f"\n[4/6] padding offset: len(clean)={len(clean_ids)} len(corrupted)={len(corrupt_ids)} "
          f"diff={diff}")
    if patch_from is not None:
        adj = patch_from - diff if diff <= 0 else patch_from
        print(f"    patch_token_from {patch_from} -> {adj} after left-pad adjustment")

    # ---- 6: readout depth (DERIVED, not assumed) -----------------------------
    print("\n[5/6] readout position (derived by generating unpatched)")
    try:
        depth = derive_lm_head_depth(llm, corrupted, answer_ids)
        check("readout depth derived", True,
              f"depth={depth} (upstream hardcodes 2, or 3 for OLMo-7B+race)")
    except RuntimeError as exc:
        check("readout depth derived", False, str(exc).splitlines()[0])
        print("    " + "\n    ".join(str(exc).splitlines()[1:]))

    # ---- 7: activation RMS at the patch site ---------------------------------
    print("\n[6/6] activation magnitude at the patch site")
    print("    (RMSNorm scale-invariance only holds when RMS^2 >> rms_norm_eps=1e-6,\n"
          "     i.e. RMS >> 1e-3. Below that, eps bites and scaling is NOT a no-op.)")
    if patch_from is not None:
        # NB: use the UNadjusted index. The left-pad offset only applies inside the
        # batched generate() where nnsight pads the shorter prompt up to the longer
        # one; a standalone trace of `clean` has no padding.
        with torch.no_grad():
            with llm.trace(clean):
                act = get_patch_site(llm, args.layer, site="auto").output[0, patch_from, :].save()
        rms = act.float().pow(2).mean().sqrt().item()
        check("RMS >> rms_norm_eps regime", rms > 1e-2,
              f"RMS={rms:.4g} at layer {args.layer}, token {patch_from} "
              f"({llm.tokenizer.decode([clean_ids[patch_from]])!r})")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("ALL CHECKS PASSED -- safe to run patching on this model.")


if __name__ == "__main__":
    main()
