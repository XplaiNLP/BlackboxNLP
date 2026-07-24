# Scale-Invariant Normalization Hides Activation Scaling: Why Patch Site Matters for Demographic Bias Localization in Clinical Text Generation

Code and data for our submission to the [BlackboxNLP 2026 Reproducibility Challenge](https://blackboxnlp.github.io/2026/reproducibility/).

We reproduce [Ahsan et al. (2025)]([https://aclanthology.org/2025.findings-emnlp.XXXX/](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-emnlp.789.pdf)) on their four models, extend to ten, and show that
two of their mechanistic findings follow from the site at which the patch is applied
rather than from the models: in post-norm architectures the patched MLP output passes
through a scale-invariant RMSNorm, so activation scaling is inert by construction.

## Repository structure

| Folder | Contents |
| --- | --- |
| `plots/` | All figures in the paper, plus the extended per-model visualizations referred to in Appendix F |
| `vignettes_outputs/` | Generated clinical vignettes per model |
| `vignettes_analysis/` | Per-model CSVs with gender, race, and refusal distributions (organized in a directory per model family), plus the scripts that produce them |
| `activation_patching/` | Patching and rewrite-score scripts |

## Setup

```bash
git clone <repo-url> && cd <repo>
pip install -r requirements.txt
```
## Reproducing the patching results

All commands are run from `activation_patching/`. Models load from the Hugging Face
Hub in 4-bit by default; `-model_name` accepts any entry in `SUPPORTED_MODELS`
(`model_registry.py`).

### 1. Localization: rewrite-score sweep

Sweeps every layer and token position and writes a layer x token pickle, used to
identify the localized layer (paper Fig. 2, Table 11).

```bash
python3 get_patching_scores.py \
    -demographic_type gender \
    -condition "multiple sclerosis" \
    -target Male \
    -model_name meta-llama/Llama-3.1-8B-Instruct \
    -output_dir outputs
```

For race, both `-source` and `-target` are required:

```bash
python3 get_patching_scores.py \
    -demographic_type race -condition "hepatitis B" \
    -source Caucasian -target Asian \
    -model_name allenai/OLMo-7B-0724-Instruct-hf -output_dir outputs
```

| Flag | Purpose |
| --- | --- |
| `-lm_head_depth` | Readout offset for the demographic token. Defaults to the per-model value in `model_registry`; overriding it reproduces the flat sweep obtained under a fixed offset. |
| `-greedy_trace` | Deterministic generation in the trace, so the score does not depend on the sampling seed. |
| `-sexed_condition` | Gender only: patch from a sex-exclusive condition (prostate cancer / preeclampsia) instead of an explicit "The patient is Male" prompt. |

The script writes `<...>-patch_scores.p` and attempts a heatmap. If plotly is absent
the pickle is still written, and the plot can be produced separately:

```bash
python3 plot_patching_scores.py -scores_path outputs/<...>-patch_scores.p
```

### 2. Flip rates: interchange accuracy

Patches the reference activation at a given layer, generates 500 vignettes per
factor at temperature 0.7, and reports the fraction matching the target
(Tables 3–6, 8).

```bash
python3 get_interchange_accuracy.py \
    -demographic_type gender \
    -condition "multiple sclerosis" \
    -target Male \
    -layer 16 \
    -model_name google/gemma-2-9b-it \
    -patch_site post_norm \
    -output_dir outputs
```

**`-patch_site` is the central variable of the paper.** `down_proj` (the default)
is the original's site and reproduces it, including the scaling failure; on the
post-norm models (Gemma-2, Gemma-3, MedGemma, OLMo-2) the script prints a warning
that the factor is annihilated by RMSNorm before the residual add. `post_norm`
patches after the normalization and restores the effect. `auto` selects per model.

| Flag | Purpose |
| --- | --- |
| `-window k` | Sliding window: patch layers `L-k … L+k` (Table 6, `SW` column). Race with `window > 0` runs factor 1 only. |
| `-alpha a [a …]` | Interpolation mode, `z := (1-a)·z_dest + a·z_src`, in place of multiplicative scaling. Changes direction rather than magnitude, so it survives RMSNorm at either site. Omit for the paper's factors 1/2/5. |
| `-prompt_id 1–10` | Run under one of the ten Zack et al. templates instead of the default patching template, testing whether the patch is a property of the model or of the prompt. |
| `-load_in_4bit false` | Full-precision bf16, to check quantization sensitivity of the unscaled patch. |
| `-outer_n`,`-inner_n` | Batch counts; the defaults (25 × 20) give the 500 vignettes reported. |

Results are written to `IA_{mode}_{prompt}_{condition}_{target}_l{L}_w{W}_{site}_{model}.csv`.

### 3. Specificity controls

Both controls reuse the same script (Tables 9, 12):

```bash
# mismatched source: neutral "patient" activation at the localized layer
python3 get_interchange_accuracy.py ... -layer 16 -mismatched_source

# random layer: the real demographic source at a non-localized layer
python3 get_interchange_accuracy.py ... -layer 20
```

The random-layer control is not a flag — it is the ordinary run with a
non-localized `-layer` (we use L20). Mismatched-source runs are tagged `MM` in the
output filename so they do not overwrite the real patch.

### 4. Fluency check

Perplexity of the patched vignettes per factor, judged by a model from a different
family than the generator. The script refuses to score a model with a judge from
its own family.

```bash
python3 get_perplexity.py -ia_csv outputs/IA_*.csv \
    -judge meta-llama/Llama-3.1-8B-Instruct
```

Perplexity should stay flat as the factor rises; a rising curve means the patch is
degrading the text. The corrupted-patch reference is ≈15.5.
