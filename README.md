# Retrieval-Augmented LLMs and Encoder Models for Multi-Label Climate Misinformation Narrative Classification

This repository contains the code and resources for our submission to **ClimateCheck@NSLP 2026 Task 2**, focusing on multi-label climate misinformation narrative classification using the CARDS taxonomy.

## Overview

Climate misinformation narratives are often expressed in complex and overlapping forms. This project investigates multiple approaches for detecting and classifying climate disinformation narratives:

- Encoder-based transformer models (ModernBERT, RoBERTa, DistilBERT)
- Decoder-only large language models (Qwen3, LLaMA)
- Prompt-enhanced instruction tuning
- Hierarchical instruction tuning
- Retrieval-augmented instruction tuning
- Targeted data augmentation strategies

Our best-performing model uses **retrieval-augmented instruction tuning with Qwen3**, achieving:

> **Macro-F1: 59.72%**  
> Second place on the official leaderboard

---

## Repository Structure
├── BERT-based_Experiments/ # Encoder-based model training scripts, Data Augmentation
├── Llama_Experiments/ # Fine-tuning on LlamMA model
├── Qwen-3_Experiments├── qwen_retrieval/ # Retrieval-augmented instruction tuning
                      ├── qwen_hierarchical/ # Hierarchical instruction tuning
                      ├── qwen_retrieval/ # Retrieval-augmented instruction tuning
├── src-Additional_Data_sets/ # Augmented_cards and CARDS datasets
├── requirements.txt
└── README.md


---

## Data

We use:

- **ClimateCheck 2026 Task 2 dataset**
- **CARDS dataset**
- **Augmented-CARDS dataset**
- Targeted augmented data (Augmented_CC26)
- Exorde dataset (for semantic similarity filtering)

All datasets follow the **CARDS hierarchical taxonomy** for narrative labels.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/XplaiNLP/ClimateCheck-NSLP-2026.git
cd ClimateCheck-NSLP-2026


Install dependencies:
pip install -r requirements.txt

---

## Contact
For questions or issues, please open a GitHub issue or contact:
  - neda.foroutan@tu-berlin.de
  - alexandra.tsiakalou@tu-berlin.de

---

## License
This repository is released under the MIT License (see LICENSE file).

