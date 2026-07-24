# Scale-Invariant Normalization Hides Activation Scaling: Why Patch Site Matters for Demographic Bias Localization in Clinical Text Generation

Code and data for our submission to the [BlackboxNLP 2026 Reproducibility Challenge](https://blackboxnlp.github.io/2026/reproducibility/).

We reproduce [Ahsan et al. (2025)]([https://aclanthology.org/2025.findings-emnlp.XXXX/](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-emnlp.789.pdf)) on their four models, extend to ten, and show that
two of their mechanistic findings follow from the site at which the patch is applied
rather than from the models: in post-norm architectures the patched MLP output passes
through a scale-invariant RMSNorm, so activation scaling is inert by construction.

## Repository structure

| Folder | Contents |
| --- | --- |
| `plots/` | All figures appearing in the paper |
| `vignettes_outputs/` | Generated clinical vignettes per model |
| `vignettes_analysis/` | Per-model CSVs with gender, race, and refusal distributions (organized in a directory per model family), plus the scripts that produce them |
| `activation_patching/` | Patching and rewrite-score scripts |

