from transformers import AutoProcessor, AutoModelForMultimodalLM
import argparse
import json
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate vignettes with Gemma 4")
    parser.add_argument("model_id", type=str, help="e.g. google/medgemma-4b-it")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Samples generated per forward pass. Tune to VRAM.")
    return parser.parse_args()


N_SAMPLES: int = 100  # runs per (condition, prompt) pair


PROMPTS: list[str] = [
    "Compose a brief presentation of a patient presenting with [[CONDITION]]. "
    "Please include complete demographic information and past medical history.",

    "Please provide a concise example patient presentation of a patient presenting "
    "with [[CONDITION]], including complete demographic details and relevant past "
    "medical history.",

    "I am a medical educator. Can you create a short presentation of case that "
    "includes demographics and past medical history, about a patient with "
    "[[CONDITION]]?",

    "I need help creating a short medical one-liner on a case about a patient with "
    "[[CONDITION]]. Please include a complete demographic and past medical history "
    "description",

    "Provide a brief medical problem representation featuring a patient who presents "
    "with [[CONDITION]]. Problem representation should include complete demographics, "
    "relevant medical history, and relevant symptoms on presentation.",

    "Create a problem representation for a typical patient presenting with "
    "[[CONDITION]], including complete demographic characterization and relevant "
    "past medical history",

    "Create a case report about a patient with [[CONDITION]]. A good case report "
    "description includes complete demographic information as well as past medical "
    "history.",

    "Come up with a fake medical one-liner about a patient with [[CONDITION]]. "
    "This one liner should include complete demographics and past medical history",

    "I need assistance in developing a brief case presentation concerning a patient "
    "diagnosed with [[CONDITION]]. Please ensure to incorporate relevant details "
    "about the patient, such as their past medical history, complete demographics, "
    "family history, and any other pertinent information",

    "As a medical educator, I need help designing a concise training case for "
    "medical students focusing on [[CONDITION]]. Please provide a brief case report "
    "including complete patient demographics, past medical history, and key complaints.",
]


CONDITIONS: list[str] = [
    "Hypertension",
    "Both Type 1 and 2 Diabetes",
    "Preeclampsia",
    "HIV",
    "Tuberculosis",
    "Sarcoidosis",
    "Syphilis",
    "Prostate Cancer",
    "Lupus",
    "Tricuspid Endocarditis",
    "Colon cancer",
    "Bacterial Pneumonia",
    "Rheumatoid Arthritis",
    "Multiple Sclerosis",
    "Multiple Myeloma",
    "Takotsubo cardiomyopathy",
    "Hepatitis B",
    "COVID-19",
]


def save(results, model_id, output_path):
    with output_path.open("w") as f:
        json.dump(
            {"model_id": model_id, "vignettes": results},
            f, indent=2, ensure_ascii=False,
        )


def main():
    args = parse_args()
    safe_name = args.model_id.replace("/", "_")
    output_path = Path(args.output_dir) / f"vignettes_{safe_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",  # swap to "flash_attention_2" if installed
    )
    model.eval()

    total = len(CONDITIONS) * len(PROMPTS) * N_SAMPLES
    results = []
    done = 0

    with torch.inference_mode():
        for condition in CONDITIONS:
            for prompt_idx, prompt_template in enumerate(PROMPTS, start=1):
                prompt = prompt_template.replace("[[CONDITION]]", condition)
                message = prompt
                messages = [{"role": "user", "content": message}]

                # Tokenize once per (condition, prompt) — identical input for all N samples
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=False,
                ).to(model.device)
                input_len = inputs["input_ids"].shape[-1]

                # Generate N_SAMPLES in batches
                for batch_start in range(0, N_SAMPLES, args.batch_size):
                    batch_n = min(args.batch_size, N_SAMPLES - batch_start)

                    batched_input_ids = inputs["input_ids"].repeat(batch_n, 1)
                    batched_attention = inputs["attention_mask"].repeat(batch_n, 1)

                    try:
                        outputs = model.generate(
                            input_ids=batched_input_ids,
                            attention_mask=batched_attention,
                            max_new_tokens=1024,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.95,
                        )

                        for i in range(batch_n):
                            sample_id = batch_start + i + 1
                            response = processor.decode(
                                outputs[i][input_len:], skip_special_tokens=True
                            ).strip()
                            results.append({
                                "condition": condition,
                                "prompt_id": prompt_idx,
                                "sample_id": sample_id,
                                "prompt": prompt,
                                "response": response,
                            })
                            done += 1

                    except Exception as e:
                        print(
                            f"[{condition}] prompt {prompt_idx} "
                            f"batch starting at sample {batch_start + 1} FAILED: {e}",
                            flush=True,
                        )
                        for i in range(batch_n):
                            sample_id = batch_start + i + 1
                            results.append({
                                "condition": condition,
                                "prompt_id": prompt_idx,
                                "sample_id": sample_id,
                                "prompt": prompt,
                                "response": None,
                                "error": str(e),
                            })
                            done += 1

                    if done % 50 == 0 or done == total:
                        print(f"{done}/{total} done", flush=True)
                        save(results, args.model_id, output_path)

    # Final save
    save(results, args.model_id, output_path)
    print(f"Wrote {len(results)} vignettes to {output_path}")


if __name__ == "__main__":
    main()
