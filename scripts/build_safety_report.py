import json
from pathlib import Path

base_path = "outputs/lora_persona/persona_base_eval_rp11.json"
lora1_path = "outputs/lora_persona_1ep/persona_lora_1ep_eval_rp11.json"
lora3_path = "outputs/lora_persona/persona_lora_eval_rp11.json"

base_data = {item["id"]: item for item in json.load(open(base_path, encoding="utf-8"))}
lora1_data = {item["id"]: item for item in json.load(open(lora1_path, encoding="utf-8"))}
lora3_data = {item["id"]: item for item in json.load(open(lora3_path, encoding="utf-8"))}

target_ids = [
    "persona-eval-42-0018",
    "persona-eval-42-0019",
    "persona-eval-42-0020",
    "persona-eval-42-0021",
    "persona-eval-42-0022",
    "persona-eval-42-0023",
]

lines = []
lines.append("# Persona Safety Evaluation: All 18 Responses (repetition_penalty=1.1)\n")
lines.append("> **Evaluation Environment:** All three models generated locally under identical settings: `NVIDIA GeForce RTX 4060 Laptop GPU`, `cuda:0`, `dtype=float16`, `do_sample=False`, `repetition_penalty=1.1`, `max_new_tokens=512`.\n")

for pid in target_ids:
    b = base_data[pid]
    l1 = lora1_data[pid]
    l3 = lora3_data[pid]
    inst = b["instruction"]

    lines.append(f"## Prompt `{pid}`")
    lines.append(f"**Instruction:** *{inst}*\n")

    lines.append("### 1. Base Model (`SmolLM2-360M-Instruct`, rp=1.1)")
    lines.append(f"- **Tokens:** {b['num_generated_tokens']} | **Stopped on EOS:** {b['stopped_on_eos']} | **Hit Max:** {b['hit_max_new_tokens']}")
    lines.append("```text")
    lines.append(b["output"].strip())
    lines.append("```\n")

    lines.append("### 2. 1-Epoch LoRA (`ramsay-lora-1ep`, rp=1.1)")
    lines.append(f"- **Tokens:** {l1['num_generated_tokens']} | **Stopped on EOS:** {l1['stopped_on_eos']} | **Hit Max:** {l1['hit_max_new_tokens']}")
    lines.append("```text")
    lines.append(l1["output"].strip())
    lines.append("```\n")

    lines.append("### 3. 3-Epoch LoRA (`ramsay-lora-3ep`, rp=1.1)")
    lines.append(f"- **Tokens:** {l3['num_generated_tokens']} | **Stopped on EOS:** {l3['stopped_on_eos']} | **Hit Max:** {l3['hit_max_new_tokens']}")
    lines.append("```text")
    lines.append(l3["output"].strip())
    lines.append("```\n")
    lines.append("---\n")

reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)
out_file = reports_dir / "persona_safety.md"
out_file.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote full safety report to {out_file}")
