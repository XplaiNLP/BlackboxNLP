"""Rewrite-score localization sweep.

Sweeps every layer and token position, patching the MLP activation from a reference
prompt (e.g. "The patient is Male") into the vignette-generation prompt, and records the
rewrite score of Hase et al. (2023): the normalized shift in the target-demographic
probability at the readout token. The output is a layer x token pickle used to identify
the localized layer.

Adapted from Ahsan et al. (2025), https://github.com/hibaahsan/interp-healthcare-bias.
Our additions are marked `EXTENSION` in-line:
  -lm_head_depth  read the score at a chosen generated-token index (default: the
                  per-model value in model_registry).
  -greedy_trace   deterministic generation in the trace, so the score is independent of
                  the sampling seed.

Example (run from activation_patching):
    python3 get_patching_scores.py -demographic_type gender -condition "multiple sclerosis" \\
        -target Male -model_name meta-llama/Llama-3.1-8B-Instruct -output_dir outputs
"""
import os
import pickle
import sys
from pathlib import Path

import argparse
import numpy as np
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_registry import (
    SUPPORTED_MODELS,
    first_content_token,
    get_layer_step,
    get_layers,
    get_lm_head_depth,
    get_text_templates,
    nnsight_logits,
)
try:
    from plot_patching_scores import save_patching_heatmap
except ImportError:  # plotly absent in headless/repro env; plotting is optional, pickle still saved
    save_patching_heatmap = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Patch gender/race")
    parser.add_argument('-demographic_type', choices=['gender', 'race'], help='demographic')
    parser.add_argument('-sexed_condition', action='store_true', help='only valid for gender. whether to patch using sexed condition')
    parser.add_argument('-condition', type=str, help='clinical condition')
    parser.add_argument('-source', type=str, help='race to flip from')
    parser.add_argument('-target', type=str, help='race/gender to flip to')
    parser.add_argument('-model_name', choices=SUPPORTED_MODELS, help='model type', default='allenai/OLMo-7B-0724-Instruct-hf')
    parser.add_argument('-output_dir', type=str)
    # EXTENSION: read the score at a chosen generated-token index; default is the
    # per-model value in model_registry.
    parser.add_argument('-lm_head_depth', type=int, default=None,
                        help='override the nnsight readout depth (default: per-model table)')
    # EXTENSION: deterministic generation in the trace (default follows the model's
    # generation_config).
    parser.add_argument('-greedy_trace', action='store_true',
                        help='greedy (deterministic) generation in the trace')



    args = parser.parse_args()
    demographic_type = args.demographic_type
    source_condition = args.condition
    model_name = args.model_name
    target = args.target.strip().lower()
    source = args.source.strip().lower() if demographic_type == 'race' else ''
    sexed_condition = True if args.sexed_condition else False
    output_dir = args.output_dir 

    if sexed_condition:
        assert demographic_type == 'gender'


    _depth_suffix = (f'-d{args.lm_head_depth}' if args.lm_head_depth is not None else '') + ('g' if args.greedy_trace else '')
    scores_path = os.path.join(output_dir, f'{demographic_type}-{source_condition}-{target}-{model_name}{_depth_suffix}-patch_scores.p'.replace('/', '-'))
    plot_path = os.path.join(output_dir, f'{demographic_type}-{source_condition}-{target}-{model_name}-patch_scores_plot.pdf'.replace('/', '-'))

    print('Demographic Type:', demographic_type)
    print('Clinical condition:', source_condition)
    print('Source:', source)
    print('Target:', target)


    text_templates = get_text_templates(model_name)
    text_template = text_templates[demographic_type]
    role = text_templates["role"]
    lm_head_depth = args.lm_head_depth if args.lm_head_depth is not None else get_lm_head_depth(model_name, demographic_type)
    print('Readout depth (lm_head_depth):', lm_head_depth, '(override)' if args.lm_head_depth is not None else '(per-model table)')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LanguageModel(model_name, torch_dtype=torch.bfloat16, device_map='auto', load_in_4bit=True,)  # torch_dtype: transformers 4.49 compatibility

    system = "You are a helpful clinical assistant."

    if demographic_type == 'gender':
        if target == 'male':
            answers = [(' Male', ' Female')]
        else:
            answers = [(' Female', ' Male')]

        if not sexed_condition:
            print('Patching using gender-explicit prompt')

            if target == 'male':
                target_condition = 'Male'
            else:
                target_condition = 'Female'

            messages = [
                {"role": role, "content": "The patient is {0}.".format(target_condition)},
            ]

        else:
            print('Patching using gender-implicit (clinical condition) prompt')
            if target == 'male':
                target_condition = 'prostate cancer'
            else:
                target_condition = 'preeclampsia'

            text = text_template.format(target_condition)

            messages = [
                        {"role": role, "content": "{0}\n\n{1}".format(system, text)},
            ]
    else:
        source = source[0].upper() + source[1:].lower()
        target = target[0].upper() + target[1:].lower()

        answers = [(' ' + target, ' ' + source)]

        target_condition = target

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

    print(prompts)

    prompt_id = 0
    corrupt_id = 1
    # EXTENSION: get_layers() handles wrappers that keep the decoder under model.language_model (Gemma-3/MedGemma)
    layers = get_layers(llm)
    N_LAYERS = len(layers)
    softmax = torch.nn.Softmax(dim=-1)

    clean_tokens = llm.tokenizer(prompts[0], return_tensors="pt")["input_ids"][0]
    corrupted_tokens = llm.tokenizer(prompts[1], return_tensors="pt")["input_ids"][0]

    clean_decoded_tokens = [llm.tokenizer.decode(token) for token in clean_tokens]
    print(clean_decoded_tokens)

    corrupted_decoded_tokens = [llm.tokenizer.decode(token) for token in corrupted_tokens]
    print(corrupted_decoded_tokens)

    target_condition_ids = llm.tokenizer(' ' + target_condition, return_tensors="pt")['input_ids'][0]
    source_condition_ids = llm.tokenizer(' ' + source_condition, return_tensors="pt")['input_ids'][0]

    diff = len(clean_tokens) - len(corrupted_tokens)
    print(diff)

    
    target_ix, source_ix = -1, -1

    if target_condition == 'prostate cancer':
        target_ix = -2  # patch 'prostate' (the sex-carrying subtoken), not 'cancer'
        print('Changing target index', target_ix)

    patch_token_from = torch.argwhere(clean_tokens == target_condition_ids[target_ix])[0][0].tolist()
    offset = 0 

    #adjust index since prompts will be padded
    if diff>0:
        offset = diff
        print('Corrupted prompt is shorter, so will be padded. Need to adjust token index by offset', offset)
        
    else:
        patch_token_from = patch_token_from - diff
        print('Clean prompt is shorter, so will be padded. Need to adjust patch_token_from index by offset', offset)
    

    print('patch_token_from', patch_token_from)
    # EXTENSION: first_content_token skips SentencePiece leading-space tokens
    answer_token_indices = [
                [first_content_token(llm.tokenizer, answers[i][j]) for j in range(2)]
                for i in range(len(answers))
        ]
    print("answer_tokens = " , answer_token_indices)
    print(answers)


    z_hs = {}
    rewrite_scores = []
    step=5

    patched_preds, corrupted_preds = [], []


    step = get_layer_step(model_name)

    for start in range(0, N_LAYERS, step):
        end = min(start + step, N_LAYERS)
        print(start, end)
    

        with torch.no_grad():
            _gen_kw = {'do_sample': False} if args.greedy_trace else {}
            with llm.generate(max_new_tokens=max(6, lm_head_depth + 2), **_gen_kw) as tracer:
                with tracer.invoke(prompts[prompt_id]) as invoker:
                    z_hs = {}
                    for layer_idx in range(N_LAYERS):
                        z = layers[layer_idx].mlp.down_proj.output
                        z_hs[layer_idx] = z[:, patch_token_from, :]

                                            
                with tracer.invoke(prompts[corrupt_id]) as invoker:
                    corrupted_logits = nnsight_logits(llm, lm_head_depth)
                        
                    corrupted_pred = corrupted_logits[0].argmax(dim=-1).save()
                    corrupted_prob = softmax(corrupted_logits[0][0])[answer_token_indices[prompt_id][0]]
                    corrupted_preds.append(corrupted_pred)
            
                for layer_idx in range(start, end):
                    for token_idx in range(len(corrupted_tokens)):
                        with tracer.invoke(prompts[corrupt_id]) as invoker:
                            z_corrupt = layers[layer_idx].mlp.down_proj.output
                            z_corrupt[:,token_idx+offset,:] = z_hs[layer_idx]
                            layers[layer_idx].mlp.down_proj.output = z_corrupt

                            patched_logits = nnsight_logits(llm, lm_head_depth)
                    
                            patched_logit_diff = (
                                                        patched_logits[0, -1, answer_token_indices[prompt_id][0]]
                                                        - patched_logits[0, -1, answer_token_indices[prompt_id][1]]
                                                    )
                                        
                            patched_pred = patched_logits[0].argmax(dim=-1).save()

                    
                            patched_prob = softmax(patched_logits[0][0])[answer_token_indices[prompt_id][0]]
                                    
                            rewrite_score = (patched_prob - corrupted_prob)/(1-corrupted_prob)
                            rewrite_scores.append(rewrite_score.save())
                            patched_result = (patched_prob - corrupted_prob)

                            patched_preds.append(patched_pred)
        

    rewrite_scores = [score.cpu().float().item() for score in rewrite_scores]
    rewrite_scores = np.array(rewrite_scores)
    rewrite_scores = np.reshape(rewrite_scores, (N_LAYERS, -1))

    corrupted_decoded_tokens = [llm.tokenizer.decode(token) for token in corrupted_tokens]
    token_labels = [f"{token}_{index}" for index, token in enumerate(corrupted_decoded_tokens)]
    layer_labels = [index for index in range(len(rewrite_scores))]

    patched_preds =  [p.cpu().float().item() for p in patched_preds]

    results = {}
    results['token_labels'] = token_labels
    results['layer_labels'] = layer_labels
    results['rewrite_scores'] = rewrite_scores
    results['patched_preds'] = patched_preds

    with open(scores_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Wrote scores pickle: {scores_path}")

    if plot_path:
        try:
            save_patching_heatmap(
                results['rewrite_scores'],
                results['token_labels'],
                results['layer_labels'],
                plot_path,
            )
        except Exception as exc:
            print(f"Plotting failed (scores pickle is still valid): {exc}")









        











