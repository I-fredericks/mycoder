"""Merge a trained LoRA adapter back into the base model (fp16) so it can be
converted to GGUF. Run on the same GPU box (or any machine with ~8GB RAM for 1.5B).

    python merge_lora.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --lora out/lora --out out/merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--lora", default="out/lora")
    ap.add_argument("--out", default="out/merged")
    args = ap.parse_args()

    # Merge into full-precision weights (not the 4-bit ones used for training).
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.lora)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
