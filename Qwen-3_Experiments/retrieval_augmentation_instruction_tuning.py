import pandas as pd
import torch
import json
import re
import sys
import requests
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

from datasets import load_dataset, Dataset
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForSeq2Seq

# Load dataset
dataset = load_dataset("rabuahmad/climatecheck")

# Keep only selected columns
dataset = dataset.select_columns(["claim", "claim_id", "abstract_id", "narrative"])

# Access train split
train_data = dataset["train"]
test_data = dataset["test"]

# Clean train data: remove repeated rows (claim) check have same narrative label
df = train_data.to_pandas()
df_test = test_data.to_pandas()

# Drop duplicates based on narrative
df_train = df.drop_duplicates()

#df = df_train.groupby('claim_id').first().reset_index()

df_train = (
    df_train
    .groupby("claim_id", as_index=False)
    .agg({
        "claim": "first",
        "narrative": lambda x: ",".join(map(str, set(x)))
    })
)

df_aug = pd.read_csv('/kaggle/input/datasets/nedafr/aug-cards/augmented_cards.csv')


cleaned_df_aug = pd.DataFrame({
    "claim": df_aug["text"],
    "narrative": np.where(
        (df_aug["acards_claim"].notna()) & (df_aug["acards_claim"] != "0_0"),
        df_aug["acards_claim"],
        pd.NA
    )
})

# Drop rows that matched neither condition
cleaned_df_aug = cleaned_df_aug.dropna(subset=["narrative"])
cleaned_df_aug = cleaned_df_aug.drop_duplicates()

dataset = load_dataset("alextsiak/cc26-cards-aug-300")

df_cc26 = dataset["train"].to_pandas()
df_cc26["narrative"] = df_cc26["narrative"].apply(lambda x: x[0])

merged_df = pd.concat(
    [df[["claim", "narrative"]],
     #cleaned_df_aug[["claim", "narrative"]]],
     #cleaned_df_cards[["claim", "narrative"]]],
     df_cc26[["claim", "narrative"]]],
    ignore_index=True
)

merged_df = merged_df.drop_duplicates().reset_index(drop=True)

Narrative_labels = ["No disinformation narrative",
                    "Global warming is not happening",
                    "Ice or permafrost or snow cover isn't melting",
                    "We're heading into an ice age or global cooling",
                    "Weather is cold or snowing",
                    "Climate hasn't warmed or changed over the last (few) decade(s)",
                    "Oceans are cooling or not warming",
                    "Sea level rise is exaggerated or not accelerating",
                    "Extreme weather isn't increasing or has happened before or isn't linked to climate change",
                    "They changed the name from 'global warming' to 'climate change'",
                    "Human greenhouse gases are not causing climate change",
                    "It's natural cycles or variation",
                    "It's non-greenhouse gas human climate forcings (aerosols, land use)",
                    "There's no evidence for greenhouse effect or carbon dioxide driving climate change",
                    "CO2 is not rising or ocean pH is not falling",
                    "Human CO2 emissions are miniscule or not raising atmospheric CO2",
                    "Climate impacts or global warming is beneficial or not bad",
                    "Climate sensitivity is low or negative feedbacks reduce warming",
                    "Species or plants or reefs aren't showing climate impacts yet or are benefiting from climate change",
                    "CO2 is beneficial or not a pollutant",
                    "It's only a few degrees (or less)",
                    "Climate change does not contribute to human conflict or threaten national security",
                    "Climate change doesn't negatively impact health",
                    "Climate solutions won't work",
                    "Climate policies (mitigation or adaptation) are harmful",
                    "Climate policies are ineffective or flawed",
                    "It's too hard to solve",
                    "Clean energy technology or biofuels won't work",
                    "People need energy (e.g., from fossil fuels or nuclear)",
                    "Climate movement or science is unreliable",
                    "Climate-related science is uncertain or unsound or unreliable (data, methods & models)",
                    "Climate movement is alarmist or wrong or political or biased or hypocritical (people or groups)",
                    "Climate change (science or policy) is a conspiracy (deception)"
                    ]

label2id = {
    "0_0": 0,
    "1_0": 1,
    "1_1": 2,
    "1_2": 3,
    "1_3": 4,
    "1_4": 5,
    "1_5": 6,
    "1_6": 7,
    "1_7": 8,
    "1_8": 9,
    "2_0": 10,
    "2_1": 11,
    "2_2": 12,
    "2_3": 13,
    "2_4": 14,
    "2_5": 15,
    "3_0": 16,
    "3_1": 17,
    "3_2": 18,
    "3_3": 19,
    "3_4": 20,
    "3_5": 21,
    "3_6": 22,
    "4_0": 23,
    "4_1": 24,
    "4_2": 25,
    "4_3": 26,
    "4_4": 27,
    "4_5": 28,
    "5_0": 29,
    "5_1": 30,
    "5_2": 31,
    "5_3": 32
}

# Reverse mapping
id2label = {v: k for k, v in label2id.items()}

from sentence_transformers import CrossEncoder
cross_model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-base",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def cross_encoder_similarity(claim, narratives, model, top_k=10):
    pairs = [(claim, n) for n in narratives]

    # scores shape: (num_narratives, 3)
    scores = model.predict(pairs)

    # Extract entailment score
    entailment_scores = scores[:, 1]

    ranked = sorted(
        zip(narratives, entailment_scores),
        key=lambda x: float(x[1]),
        reverse=True
    )

    return [
        {"narrative": n, "score": float(s)}
        for n, s in ranked[:top_k]
    ]

def sim_narr(df_set):
    multi_predictions = []
    prediction_score = []
    for i, claim in enumerate(df_set["claim"]):  # train_narr
      results = cross_encoder_similarity(claim, Narrative_labels, cross_model, top_k=10)

      pred_labels= {}
      score_labels = []
      for r in results:
        index = Narrative_labels.index(r['narrative'])
        pred_label = id2label.get(index)
        pred_labels[pred_label] = r['narrative']
        #score_labels.append(r['score'])
        #print(r['narrative'], "index=", index, "pred_labels =", id2label.get(index), )

      multi_predictions.append(pred_labels)
      prediction_score.append(score_labels)
    return multi_predictions

train_sim_narratives = sim_narr(merged_df)
train_sim_str = ["\n".join([f"{k}: {v}" for k, v in sim.items()]) for sim in train_sim_narratives]

test_sim_narratives = sim_narr(df_test)
test_sim_str = ["\n".join([f"{k}: {v}" for k, v in sim.items()]) for sim in test_sim_narratives]

Narratives_list = {
    "0_0": "No disinformation narrative",
    "1_0": "Global warming is not happening",
    "1_1": "Ice/permafrost/snow cover isn't melting",
    "1_2": "We're heading into an ice age/global cooling",
    "1_3": "Weather is cold/snowing",
    "1_4": "Climate hasn't warmed/changed over the last (few) decade(s)",
    "1_5": "Oceans are cooling/not warming",
    "1_6": "Sea level rise is exaggerated/not accelerating",
    "1_7": "Extreme weather isn't increasing/has happened before/isn't linked to climate change",
    "1_8": "They changed the name from 'global warming' to 'climate change'",
    "2_0": "Human greenhouse gases are not causing climate change",
    "2_1": "It's natural cycles/variation",
    "2_2": "It's non-greenhouse gas human climate forcings (aerosols, land use)",
    "2_3": "There's no evidence for greenhouse effect/carbon dioxide driving climate change",
    "2_4": "CO2 is not rising/ocean pH is not falling",
    "2_5": "Human CO2 emissions are miniscule/not raising atmospheric CO2",
    "3_0": "Climate impacts/global warming is beneficial/not bad",
    "3_1": "Climate sensitivity is low/negative feedbacks reduce warming",
    "3_2": "Species/plants/reefs aren't showing climate impacts yet/are benefiting from climate change",
    "3_3": "CO2 is beneficial/not a pollutant",
    "3_4": "It's only a few degrees (or less)",
    "3_5": "Climate change does not contribute to human conflict/threaten national security",
    "3_6": "Climate change doesn't negatively impact health",
    "4_0": "Climate solutions won't work",
    "4_1": "Climate policies (mitigation or adaptation) are harmful",
    "4_2": "Climate policies are ineffective/flawed",
    "4_3": "It's too hard to solve",
    "4_4": "Clean energy technology/biofuels won't work",
    "4_5": "People need energy (e_g_, from fossil fuels/nuclear)",
    "5_0": "Climate movement/science is unreliable",
    "5_1": "Climate-related science is uncertain/unsound/unreliable (data, methods & models)",
    "5_2": "Climate movement is alarmist/wrong/political/biased/hypocritical (people or groups)",
    "5_3": "Climate change (science or policy) is a conspiracy (deception)"
}

Narratives_list_str = "\n".join([f"{k}: {v}" for k, v in Narratives_list.items()])

def build_prompt(claim, Similar_Narratives_str, Narratives_list_str):
    return f"""You are an expert system for detecting climate change disinformation.

You will be given a single claim and a list of 10 narrative labels selected from the full narrative inventory because they are most semantically similar to the claim.

Your task:
1. Carefully read the claim.
2. Review the provided Similar_Narratives list.
3. Classify the claim by assigning the most appropriate narrative ID(s) from the provided Similar_Narratives list and Full Narrative Inventory.

Instructions:
- Select the narrative ID(s) whose description best matches the claim.
- You MUST choose ONLY from the provided Similar_Narratives.
- If the claim clearly aligns with more than one narrative, return all applicable narrative IDs separated by a semicolon (;).
- If the claim does NOT contain climate change disinformation or does not match any listed narrative, return exactly: 0_0.
- Do NOT explain your reasoning.
- Do NOT output anything other than the narrative ID(s).

Full Narrative Inventory:
{Narratives_list_str}

Now classify the following claim:

Claim: "{claim}"
Similar_Narratives: {Similar_Narratives_str}
Narrative IDs: """

claims = merged_df["claim"].tolist()
narratives = merged_df["narrative"].tolist()

messages_all = []

for i in range(len(claims)):
    claim = claims[i]
    codes = narratives[i]
    train_sim = train_sim_str[i]

    prompt = build_prompt(claim, train_sim, Narratives_list_str)
    #prompt = build_prompt(claim, Narratives_list_str)

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": codes}
    ]

    messages_all.append({"messages": messages})

max_seq_length = 8192
load_in_4bit = True
dtype = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

messages_data = messages_all
dataset = Dataset.from_list(messages_data)


tokenizer = get_chat_template(
    tokenizer,
    #chat_template = "qwen-2.5",
    chat_template = "qwen-3",

)

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        ) for messages in examples["messages"]
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

class SafeSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

trainer = SafeSFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        save_strategy = "steps",
        save_steps = 500,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

print(trainer.train_dataset[5]["labels"])

trainer_stats = trainer.train()

def predict_claims(claims, sim, max_new_tokens=16):
    results = []
    model.eval()

    for i in range(len(claims)):
        prompt = build_prompt(claims[i], sim[i], Narratives_list_str)

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Optional: extract only the last line
        results.append(decoded.strip())

        del inputs, outputs
        torch.cuda.empty_cache()

    return results

def extract_narrative_ids(text):
    match = re.search(r"(?:^|\n)(\d_\d(?:;\d_\d)*)\s*$", text.strip())
    return match.group(1) if match else None

test_predictions = predict_claims(df_test["claim"].tolist(), test_sim_str)
test_extracted_predictions = [
    extract_narrative_ids(pred) for pred in test_predictions
]

train_predictions = predict_claims(df_train["claim"][0:100].tolist(), train_sim_narratives[0:100])
train_extracted_predictions = [
    extract_narrative_ids(pred) for pred in train_predictions
]

df_submission = df_test[["claim_id", "abstract_id", "narrative"]].copy()
df_submission["narrative"] = Real_test_extracted_predictions

df_submission.to_csv("predictions.csv", index=False)