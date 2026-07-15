# BlackboxNLP 

This repository hosts the code for our BlackboxNLP 2026 Reproducibility Challenge submission [title tbd]


## Use


1. To generate vignettes (eg. using MedGemma-4B):


```bash
python vignette_generation/vignette_gemma.py google/medgemma-4b-it --output-dir <path/to/output> --batch-size <int>
```


2. To analyze the gender and race distribution of the vignettes:


```bash
python inspect-vignettes.py --vignettes <path/to/vignettes.json> --out_dir <path/to/output> [--use_llm_refusal_check]
```

3. To run the activation patching:

```
```
