import sys
import torch
from pathlib import Path
from src.align.generate import generate_on_eval_set

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Running unified persona generation on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

eval_set = "eval_sets/persona_style_eval.json"

# 1. Base model
print("\n--- 1. Generating Base SmolLM2 rp=1.1 ---")
generate_on_eval_set(
    model_name_or_path="HuggingFaceTB/SmolLM2-360M-Instruct",
    eval_set_path=eval_set,
    output_path="outputs/lora_persona/persona_base_eval_rp11.json",
    generator_name="smollm2-360m-instruct-base-rp11",
    dataset_name="persona_style_eval",
    max_new_tokens=512,
    batch_size=8,
    repetition_penalty=1.1,
    no_repeat_ngram_size=0,
    use_chat_template=True,
    lora_adapter_path=None,
)

# 2. 1-epoch LoRA
print("\n--- 2. Generating 1-epoch LoRA rp=1.1 ---")
generate_on_eval_set(
    model_name_or_path="HuggingFaceTB/SmolLM2-360M-Instruct",
    eval_set_path=eval_set,
    output_path="outputs/lora_persona_1ep/persona_lora_1ep_eval_rp11.json",
    generator_name="smollm2-360m-instruct-ramsay-lora-1ep-rp11",
    dataset_name="persona_style_eval",
    max_new_tokens=512,
    batch_size=8,
    repetition_penalty=1.1,
    no_repeat_ngram_size=0,
    use_chat_template=True,
    lora_adapter_path="outputs/lora_persona_1ep/best_adapter",
)

# 3. 3-epoch LoRA
print("\n--- 3. Generating 3-epoch LoRA rp=1.1 ---")
generate_on_eval_set(
    model_name_or_path="HuggingFaceTB/SmolLM2-360M-Instruct",
    eval_set_path=eval_set,
    output_path="outputs/lora_persona/persona_lora_eval_rp11.json",
    generator_name="smollm2-360m-instruct-ramsay-lora-dpo-rp11",
    dataset_name="persona_style_eval",
    max_new_tokens=512,
    batch_size=8,
    repetition_penalty=1.1,
    no_repeat_ngram_size=0,
    use_chat_template=True,
    lora_adapter_path="outputs/lora_persona/best_adapter",
)

print("\nUnified persona rp=1.1 generation completed!")
