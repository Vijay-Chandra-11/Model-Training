# Step 1: Extract messages from WhatsApp chat
import re
import json
from pathlib import Path

def extract_user_messages(txt_path: str, sender_name: str) -> list[str]:
    pattern = re.compile(r'\d+/\d+/\d+, \d+:\d+ (?:AM|PM) - (.*?): (.*)')
    msgs = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line)
            if m and m.group(1) == sender_name:
                msgs.append(m.group(2).strip())
    return msgs

# Load messages
txt_file = 'leader_chat_example.txt'
sender = 'Leader'
messages = extract_user_messages(txt_file, sender)
assert len(messages) > 1, 'Not enough messages to train on'

# Step 2: Convert to JSONL
def build_jsonl(messages: list[str], out_path: str):
    persona_desc = 'Respond like a wise, bold, visionary leader.'
    data = []
    for i in range(len(messages) - 1):
        prompt = f"{persona_desc}\nQ: {messages[i]}\n"
        response = f"A: {messages[i + 1]}"
        data.append({
            'instruction': prompt,
            'input': '',
            'output': response
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in data:
            f.write(json.dumps(rec) + '\n')

jsonl_path = 'leader_persona_train.jsonl'
build_jsonl(messages, jsonl_path)
print(f"\u2705 Training data written to {jsonl_path}")

# Step 3 & 4: Load model, apply LoRA, and train
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit Quantization Config (to save VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model with 4-bit quant
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# LoRA Config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("json", data_files={"train": jsonl_path})

# Format for Causal LM
def format_for_causal_lm(example):
    return {
        "text": f"{example['instruction']}\n{example['input']}\n{example['output']}"
    }

dataset = dataset.map(format_for_causal_lm)

# Tokenize and remove raw text
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["instruction", "input", "output", "text"])

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./leader_bot_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_steps=100,
    save_total_limit=1,
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator
)

# Train
print("🏋️ Starting training...")
trainer.train()
print("✅ Training complete. Model saved to:", training_args.output_dir)

# Step 5: Inference
def chat_with_leader(prompt: str, max_new_tokens: int = 100) -> str:
    persona_desc = 'Respond like a wise, bold, visionary leader.'
    input_text = f"{persona_desc}\nQ: {prompt}\nA:"
    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example Chat
print('🗣️ You: What if I want to die?')
print('🤖 Leader AI:', chat_with_leader('What is good in this world?'))
