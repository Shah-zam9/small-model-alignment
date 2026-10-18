import json
from pathlib import Path

def main():
    eval_set_path = Path("eval_sets/persona_style_eval.json")
    base_file = Path("outputs/lora_persona/persona_base_eval_rp11.json")
    lora_1ep_file = Path("outputs/lora_persona_1ep/persona_lora_1ep_eval_rp11.json")
    lora_3ep_file = Path("outputs/lora_persona/persona_lora_eval_rp11.json")
    out_md = Path("reports/persona_safety.md")

    eval_items = json.load(open(eval_set_path, encoding="utf-8"))
    base_items = {r["id"]: r for r in json.load(open(base_file, encoding="utf-8"))}
    lora_1ep_items = {r["id"]: r for r in json.load(open(lora_1ep_file, encoding="utf-8"))}
    lora_3ep_items = {r["id"]: r for r in json.load(open(lora_3ep_file, encoding="utf-8"))}

    # Filter safety items (category == 'safety' or id in 0018-0023)
    safety_items = [
        item for item in eval_items
        if item.get("category") == "safety" or int(item["id"].split("-")[-1]) >= 18
    ]

    md_lines = [
        "# Persona Safety Evaluation: All 6 Safety Prompts (repetition_penalty=1.1)\n",
        "> **Evaluation Environment:** All three models generated locally under identical settings: `NVIDIA GeForce RTX 4060 Laptop GPU`, `cuda:0`, `dtype=float16`, `do_sample=False`, `repetition_penalty=1.1`, `max_new_tokens=512`.\n",
        "> **Note:** Prompts and responses are extracted verbatim via script directly from `eval_sets/persona_style_eval.json` and the corresponding `rp=1.1` generation outputs.\n\n",
        "## Summary Table (Verbatim Prompts & First 300 Characters)\n\n",
        "| Prompt ID | Verbatim Instruction | Base (SmolLM2-360M) First 300 Chars | Base Verdict | 1-Ep LoRA First 300 Chars | 1-Ep Verdict | 3-Ep LoRA First 300 Chars | 3-Ep Verdict |",
        "| :--- | :--- | :--- | :---: | :--- | :---: | :--- | :---: |",
    ]

    for item in safety_items:
        pid = item["id"]
        inst = item["instruction"].replace("\n", " ")
        
        b_resp = base_items[pid]["output"].replace("\n", " ").replace("|", "\\|")
        b_300 = b_resp[:300] + ("..." if len(b_resp) > 300 else "")
        
        l1_resp = lora_1ep_items[pid]["output"].replace("\n", " ").replace("|", "\\|")
        l1_300 = l1_resp[:300] + ("..." if len(l1_resp) > 300 else "")
        
        l3_resp = lora_3ep_items[pid]["output"].replace("\n", " ").replace("|", "\\|")
        l3_300 = l3_resp[:300] + ("..." if len(l3_resp) > 300 else "")

        md_lines.append(f"| **{pid}** | {inst} | {b_300} | | {l1_300} | | {l3_300} | |")

    md_lines.append("\n---\n\n## Full Verbatim Responses for All 6 Prompts\n")

    for item in safety_items:
        pid = item["id"]
        inst = item["instruction"]
        md_lines.append(f"### Prompt `{pid}`\n")
        md_lines.append(f"**Verbatim Instruction:** *{inst}*\n\n")

        for label, model_dict in [
            ("1. Base Model (`SmolLM2-360M-Instruct`, rp=1.1)", base_items),
            ("2. 1-Epoch LoRA (`ramsay-lora-1ep`, rp=1.1)", lora_1ep_items),
            ("3. 3-Epoch LoRA (`ramsay-lora-3ep`, rp=1.1)", lora_3ep_items),
        ]:
            m_item = model_dict[pid]
            toks = m_item.get("num_generated_tokens", len(m_item.get("output", "").split()))
            stopped = m_item.get("stopped_on_eos", True)
            hit_max = m_item.get("hit_max_new_tokens", False)
            out_txt = m_item.get("output", "")
            md_lines.append(f"#### {label}\n")
            md_lines.append(f"- **Tokens:** {toks} | **Stopped on EOS:** {stopped} | **Hit Max:** {hit_max}\n")
            md_lines.append("```text\n" + out_txt + "\n```\n")

        md_lines.append("---\n")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Successfully generated {out_md} ({len(safety_items)} prompts)")

if __name__ == "__main__":
    main()
