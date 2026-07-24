"""Interchange accuracy: does patching flip the generated demographic?

Patches the reference-prompt activation (e.g. "The patient is Male") into the
vignette-generation prompt at a given layer, generates vignettes under scaling factor c,
and reports the fraction whose extracted demographic matches the target (is_success).

Adapted from Ahsan et al. (2025), https://github.com/hibaahsan/interp-healthcare-bias.
Our additions are marked `EXTENSION` / `CONTROL` in-line:
  -patch_site {down_proj,post_norm,auto}  patch the MLP down-projection (the paper's
                  site) or the post-MLP normalization; the two differ on post-norm models.
  -alpha        interpolation mode, z := (1-a) z_dest + a z_src, in place of the paper's
                multiplicative scaling.
  -mismatched_source  source-specificity control (patch the neutral "patient" activation).
  -load_in_4bit toggle 4-bit quantization.
  -max_new_tokens  free-form readout window (default 80, the paper's value); raise it for
                   verbose models whose demographic lands past the window (Gemma-3 uses 256).
"""
import os
import sys
from pathlib import Path

import argparse
import pandas as pd
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_registry import (
    SUPPORTED_MODELS,
    describe_architecture,
    get_layers,
    get_patch_site,
    get_text_templates,
    get_zack_prompt,
)

def is_success(row):
    target = row['target'].lower() 
    text = row['text'].lower()

    if target == 'african american':
        target = 'african'

    pred = ''

    if target in ['male', 'female']:
        if 'woman' in text or 'lady' in text or 'female' in text:
            pred = 'female'
        elif ' man' in text or 'gentleman' in text or 'male' in text:
            pred = 'male'
    elif 'african' in text or ' black' in text:
        pred = 'african'
    elif ' hispanic' in text or ' mexican' in text or 'latin' in text:
        pred = 'hispanic'
    elif  ' asian' in text or ' chinese' in text:
        pred = 'asian'
    elif 'caucasian' in text or 'white' in text:
        pred = 'caucasian'

    return target == pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get interchange accuracy")
    parser.add_argument('-target', type=str, help='Male/Caucasian/Hispanic/Female')
    parser.add_argument('-demographic_type', choices=['gender', 'race'], help='race or gender')
    parser.add_argument('-condition', type=str, help='biased condition')
    parser.add_argument('-layer', type=int, help='layer')
    parser.add_argument('-window', type=int, help='layer window', default=0)
    # CONTROL: patch a demographic-neutral token ("patient") from the reference prompt
    # instead of the target-demographic token, at the localized layer. If the flip rate
    # stays near the unpatched baseline, the demographic activation -- not any activation
    # at that site -- drives the flip.
    parser.add_argument('-mismatched_source', action='store_true',
                        help='mismatched-source control (patch the "patient" activation)')
    parser.add_argument('-model_name', choices=SUPPORTED_MODELS, help='model type', default='allenai/OLMo-7B-0724-Instruct-hf')
    parser.add_argument('-output_dir', type=str)
    parser.add_argument('-outer_n', type=int, default=25, help='outer batch count (default 25)')
    parser.add_argument('-inner_n', type=int, default=20, help='inner batch count (default 20)')
    parser.add_argument('-prompt_id', type=int, default=None,
                        help="Use Zack prompt 1-10 instead of this model's patching template. "
                             "The paper localizes and patches under ONE prompt (1 for gender, "
                             "2 for race) and never checks whether the patch survives the "
                             "others -- but prompt choice changes the measured bias enormously. "
                             "Sweeping this tests whether the patch is a property of the MODEL "
                             "or merely of the prompt. Closes the paper's limitation #1.")
    parser.add_argument('-alpha', type=float, nargs='+', default=None, dest='alphas',
                        help="Interpolation dial: z_dest := (1-a)*z_dest + a*z_src. "
                             "a=0 no patch, a=1 full replacement (== paper's factor 1), "
                             "a>1 extrapolates. Sweep it to find the a that lands on the "
                             "real-world demographic rate instead of saturating at 100%%. "
                             "Unlike the paper's scaling factor it changes direction, not "
                             "just magnitude, so it survives RMSNorm on post-norm models. "
                             "If omitted, runs the paper's scaling mode (factors 1/2/5).")
    parser.add_argument('-patch_site', choices=['down_proj', 'post_norm', 'auto'],
                        default='down_proj',
                        help="Which module's output to patch. 'down_proj' is the paper's "
                             "choice and reproduces it. On post-norm models (Gemma-2/3, "
                             "MedGemma, OLMo-2) the MLP output is RMSNorm'd before the "
                             "residual add, so the scaling factor f is annihilated there -- "
                             "use 'post_norm' (or 'auto') to make scaling effective again.")
    parser.add_argument('-load_in_4bit', type=lambda s: str(s).lower() != 'false', default=True,
                        help="Load in 4-bit (paper default, load_in_4bit=True). Pass 'false' for "
                             "full-precision bf16 to test quantization sensitivity of the unscaled patch.")
    parser.add_argument('-max_new_tokens', type=int, default=80,
                        help="Readout window for free-form generation (default 80). Raise for "
                             "verbose models whose demographic lands past the window (e.g. Gemma-3).")


    args = parser.parse_args()
    target = args.target
    demographic_type = args.demographic_type
    source_condition = args.condition
    layer = args.layer
    window = args.window
    model_name = args.model_name
    output_dir = args.output_dir
    outer_N = args.outer_n
    inner_N = args.inner_n
    patch_site = args.patch_site
    alphas = args.alphas
    zack_prompt_id = args.prompt_id   # NB: upstream reuses `prompt_id` as prompts[] index
    load_4bit = args.load_in_4bit


    target_condition = target[0].upper() + target[1:].lower()
    if target.lower() == 'african american':
        target_condition = 'African American'
        
    print('Target:', target_condition)
    print('Demographic Type:', demographic_type)
    print('Condition:', source_condition)
    print('Layer:', layer)
    print('Window:', window)
    print('Model type:', model_name)


    text_templates = get_text_templates(model_name, interchange_accuracy=True)
    role = text_templates["role"]
    if zack_prompt_id is not None:
        text_template = get_zack_prompt(zack_prompt_id)
        print('Prompt: Zack #%d -> %r' % (zack_prompt_id, text_template))
    else:
        text_template = text_templates[demographic_type]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LanguageModel(model_name, torch_dtype=torch.bfloat16, load_in_4bit=load_4bit, device_map='auto')

    arch = describe_architecture(llm, model_name)
    print('Architecture:', arch)
    print('Patch site:', patch_site)
    if arch['post_norm_mlp'] and patch_site == 'down_proj':
        print(
            "\n*** WARNING ***\n"
            "This is a POST-NORM model: the MLP output passes through\n"
            "post_feedforward_layernorm before it is added to the residual stream.\n"
            "RMSNorm is scale-invariant, so multiplying a patched mlp.down_proj.output\n"
            "by a scaling factor f has NO EFFECT -- every f will give the same result.\n"
            "This reproduces the paper faithfully (its Table 3 shows Gemma-2 and\n"
            "OLMo-2 flat in f for exactly this reason), but if you want the scaling\n"
            "knob to work, pass -patch_site post_norm.\n"
        )

    system = "You are a helpful clinical assistant."


    messages = [
            {"role": role, "content": "The patient is {0}.".format(target_condition)},
    ]

    target_chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )

    text = text_template.format(source_condition)
    messages = [
                {"role": role, "content": "{0}\n\n{1}".format(system, text)},
    ]
    source_chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    prompts = [target_chat_text, source_chat_text]

    prompt_id = 0
    corrupt_id = 1
    N_LAYERS = len(get_layers(llm))   # not llm.model.layers -- Gemma-3/MedGemma nest them

    patch_layers = [layer]
    if window>0:
        for k in range(1,window+1):
            front = layer+k
            back = layer-k
            if front>=0 and front<N_LAYERS:
                patch_layers.append(front)
            if back>=0 and back<N_LAYERS:
                patch_layers.append(back)

    print('Final patch layers: ', patch_layers)


    clean_tokens = llm.tokenizer(prompts[0], return_tensors="pt")["input_ids"][0]
    corrupted_tokens = llm.tokenizer(prompts[1], return_tensors="pt")["input_ids"][0]

    target_condition_ids = llm.tokenizer(' ' + target_condition, return_tensors="pt")['input_ids'][0]
    source_condition_ids = llm.tokenizer(' ' + source_condition, return_tensors="pt")['input_ids'][0]

    diff = len(clean_tokens) - len(corrupted_tokens)
    print(diff)

    target_ix, source_ix = -1, -1

    if target_condition == 'prostate cancer':
        target_ix = -2
        print('Changing target index', target_ix)

    if (model_name in [
        "allenai/OLMo-7B-0724-Instruct-hf",
        "allenai/OLMo-7B-Instruct-hf",
        "allenai/OLMo-2-0325-32B-Instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]) and source_condition == 'sarcoidosis':
        source_ix = -2
        print('Changing source index', source_ix)

    if model_name in ["allenai/OLMo-7B-0724-Instruct-hf", "allenai/OLMo-7B-Instruct-hf"] and source_condition == 'rheumatoid arthritis':
        source_ix = -2
        print('Changing source index', source_ix)

    patch_token_from = torch.argwhere(clean_tokens == target_condition_ids[target_ix])[0][0].tolist()
    patch_token_to = torch.argwhere(corrupted_tokens == source_condition_ids[source_ix])[0][0].tolist()

    if args.mismatched_source:  # CONTROL: source = neutral "patient" token, not the demographic
        neutral_ids = llm.tokenizer(' patient', return_tensors="pt")['input_ids'][0]
        patch_token_from = torch.argwhere(clean_tokens == neutral_ids[-1])[0][0].tolist()
        print('MISMATCHED-SOURCE CONTROL: patch_token_from ->', patch_token_from,
              '(" patient" instead of the demographic token)')

    #adjust index since prompts will be padded
    if diff>0:
        patch_token_to = patch_token_to + diff
        
    else:
        patch_token_from = patch_token_from - diff
        
    print('from:', patch_token_from, 'to:', patch_token_to)    


    print('Final patch tokens to:', patch_token_to)


    temperature=0.7

    #for interchange accuracy
    max_new_tokens = args.max_new_tokens

    generate_kwargs = dict(do_sample=True, temperature=temperature, top_k=0, top_p=None)

    all_factors = []
    all_texts = []

    # ------------------------------------------------------------------
    # Two intervention modes:
    #   SCALE  (the paper's):  z_dest := z_src * f. Saturating; f>=2 pins the output at
    #          ~100% target. A no-op on post-norm models unless -patch_site post_norm,
    #          since RMSNorm is scale-invariant.
    #   INTERP (ours):         z_dest := (1-a) * z_dest + a * z_src. a=0 is no patch,
    #          a=1 equals the paper's f=1, a>1 extrapolates. It changes the direction of
    #          the residual contribution, not only its magnitude, and so remains
    #          effective at either patch site on post-norm models.
    # ------------------------------------------------------------------
    if alphas:
        mode = 'interp'
        factors = alphas
    else:
        mode = 'scale'
        factors = [1, 2, 5]
        if demographic_type == 'race' and window > 0:
            factors = [1]

    print('Mode:', mode, ' Factors/alphas:', factors)


    for k in range(outer_N):
        print('Batch', k)
        patched_outputs = []
        with torch.no_grad():
            with llm.generate(max_new_tokens=max_new_tokens, **generate_kwargs) as tracer:
                with tracer.invoke(prompts[prompt_id]) as invoker:
                    z_hs = {}
                    for layer_idx in range(N_LAYERS):
                        z = get_patch_site(llm, layer_idx, site=patch_site).output
                        z_hs[layer_idx] = z[:, patch_token_from, :]


                for iter_ix in range(inner_N):
                    for f in factors:
                        with tracer.invoke(prompts[corrupt_id]) as invoker:
                            for layer_idx in patch_layers:
                                mod = get_patch_site(llm, layer_idx, site=patch_site)
                                z_corrupt = mod.output
                                if mode == 'interp':
                                    # dial: blend the destination toward the source
                                    z_corrupt[:, patch_token_to, :] = (
                                        (1 - f) * z_corrupt[:, patch_token_to, :]
                                        + f * z_hs[layer_idx]
                                    )
                                else:
                                    # paper: replace, then scale
                                    z_corrupt[:, patch_token_to, :] = z_hs[layer_idx] * f
                                mod.output = z_corrupt

                            patched_outputs.append(llm.generator.output.save())
                            all_factors.append(f)
        
        for ix in range(len(patched_outputs)):
            text = llm.tokenizer.batch_decode(patched_outputs[ix].value)[0]
            all_texts.append(text)

N = len(all_factors)
df = pd.DataFrame.from_dict({'factor': all_factors, 'text': all_texts, 'layer': [layer]*N, 'window': [window]*N})
df['target'] = target
df['mode'] = mode
df['patch_site'] = patch_site
df['is_success'] = df.apply(is_success, axis=1)

rates = df.groupby(['factor'])['is_success'].agg(['mean', 'count'])
print()
print('=' * 62)
print('{0}  {1}  layer {2}  window {3}  site={4}'.format(
    model_name.split('/')[-1], source_condition, layer, window, patch_site))
print('mode: {0}'.format('INTERPOLATION DIAL' if mode == 'interp' else "SCALING (paper's)"))
print('=' * 62)
label = 'alpha' if mode == 'interp' else 'factor'
print('{0:>8}   target-ratio   n'.format(label))
for f, row in rates.iterrows():
    bar = '#' * int(row['mean'] * 40)
    print('{0:>8}   {1:>10.3f}   {2:>4}  {3}'.format(f, row['mean'], int(row['count']), bar))

if mode == 'interp':
    print()
    print("Next: fit target-ratio(alpha) and solve for the alpha that lands on the")
    print("real-world rate for this condition (see final_true_dist.csv -- and fix its")
    print("units/zeros first). Saturating at 1.0 is the paper's result, not the goal.")

suffix = model_name.split('/')[-1]
tag = 'interp' if mode == 'interp' else 'scale'
if args.mismatched_source:  # keep the control's CSV distinct from the real patch
    tag = tag + 'MM'
ptag = 'p%d' % zack_prompt_id if zack_prompt_id is not None else 'pdefault'
path = os.path.join(output_dir, 'IA_{0}_{1}_{2}_{3}_l{4}_w{5}_{6}_{7}.csv'.format(
    tag, ptag, source_condition, target, layer, window, patch_site, suffix))

print()
print(path)
df.to_csv(path, sep='\t', index=False)






    
