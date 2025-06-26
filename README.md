# 🧠 Leader Bot — Visionary Persona Fine-Tuned Model

This project fine-tunes a quantized **Mistral-7B-Instruct** model using **LoRA** to simulate a bold, wise, and visionary leader. It learns from WhatsApp-style motivational conversations to produce inspiring, leadership-driven replies.

---

## 🚀 Features

- ✅ Extracts messages from real WhatsApp chats
- ✅ Converts data to instruction-style format for fine-tuning
- ✅ Uses Hugging Face + PEFT to fine-tune in 4-bit precision
- ✅ Outputs a leader-like chatbot capable of motivational guidance

---

## 🧪 Example Chat

```txt
🗣️ You: I’m nervous about presenting tomorrow.
🤖 Leader AI: Nervous energy is the precursor to greatness. Channel it into your presentation.
```

---

## 🧠 Model Details

- **Base Model**: mistralai/Mistral-7B-Instruct-v0.2

- **Quantization**: 4-bit (nf4, float16 compute)

- **Fine-Tuning**: LoRA (PEFT)

- **Persona**: Bold, visionary leadership style

---

## 📈 Output Location

- Fine-tuned checkpoints are saved to:  **./leader_bot_model/**

---

## 🧰 Tools and Environment

    Jupyter Notebook: For writing and executing Python code during the training process.

    Python 3.11

    PyTorch: Deep learning framework used for model building and training.

    Hugging Face Transformers: For model architecture and tokenizer support.

    Datasets library: For dataset loading and handling.

    Accelerate and PEFT: For optimizing and fine-tuning large language models.

    bitsandbytes: For memory-efficient training (quantization support).
