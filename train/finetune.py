"""QLoRA fine-tune of a small open coding model. Run on a GPU (Colab/Kaggle/RunPod).

Colab setup cell:
    !pip install -q "transformers>=4.46" "trl>=0.12" "peft>=0.13" "bitsandbytes>=0.44" datasets accelerate

Run:
    python finetune.py --train data/train.jsonl --val data/val.jsonl --out out/lora

Notes:
 - A free Colab T4 (16GB) handles a 1.5B model comfortably, a 7B with 4-bit + short context.
 - TRL/PEFT APIs shift between versions. If an argument errors, check the TRL docs
   for SFTConfig / SFTTrainer in your installed version.
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="out/lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto"
    )

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    data = load_dataset("json", data_files={"train": args.train, "validation": args.val})

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        max_length=args.max_len,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tok,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"LoRA adapter saved to {args.out}")


if __name__ == "__main__":
    main()
