# REBOOT: Fine-Tuning Qwen-7B for Multi-Role NPC Persona Dialogue

> 🤖 End-to-end LLM fine-tuning pipeline using **LoRA / QLoRA** on **Qwen 2.5 7B**, designed for specialized game characters and multi-persona conversational agents with Discord & Unreal Engine integration.

---

## 📌 Project Overview
This project fine-tunes the open-weights **Qwen 2.5 7B** model on customized character dialogue datasets. It explores parameter-efficient fine-tuning (PEFT) to adapt foundation LLMs into distinct conversational personas (`Nemesis` & `Prometheus`), while maintaining low latency and reasoning capabilities.

---

## 🏗️ Architecture & Modules
The repository is structured into modular stages covering the complete machine learning lifecycle:
- **`00_finetune/`**: Training scripts, hyperparameters, and GPU execution configuration.
- **`01_lora/`**: LoRA adapter weights, PEFT configuration, and checkpoint exports.
- **`02_DC/`**: Multi-bot Discord client bridging user voice/text interactions with model inference.
- **`03_UE/`**: Game engine interop bridge for dynamic Unreal Engine NPC dialogue dispatch.
- **`04_CLI/`**: Interactive command-line interface for local model evaluation and prompt testing.

---

## 🛠️ Tech Stack
- **Foundation Model:** Qwen 2.5 7B Instruct
- **PEFT / Acceleration:** LoRA, QLoRA, Hugging Face `peft`, `transformers`, `bitsandbytes`
- **Application Bridges:** Discord.py, REST / Webhooks, Unreal Engine HTTP Bridge
- **Environment:** PyTorch, Python 3.10+

---

## 🚀 Quick Evaluation (CLI)
To test the persona generation directly in the terminal:
```bash
python 04_CLI/chat.py --model_path ./01_lora --persona nemesis
```

---

## 📄 License
This repository is released under the MIT License.
