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

df = (
    df_train
    .groupby("claim_id", as_index=False)
    .agg({
        "claim": "first",
        "narrative": lambda x: ",".join(map(str, set(x)))
    })
)

len(df_aug)

df_aug = pd.read_csv('/augmented_cards.csv')


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

df_cards_train = pd.read_csv('/cards/training.csv')
df_cards_valid = pd.read_csv('/cards/validation.csv')
df_cards_test = pd.read_csv('/cards/test.csv')

merged_df_cards = pd.concat(
    [df_cards_train[["text", "claim"]],
     df_cards_valid[["text", "claim"]]],
     #df_cards_test[["text", "claim"]]],
    ignore_index=True
)

cleaned_df_cards = pd.DataFrame({
    "claim": merged_df_cards["text"],
    "narrative": np.where(
        (merged_df_cards["claim"].notna()) & (merged_df_cards["claim"] != "0_0"),
        merged_df_cards["claim"],
        pd.NA
    )
})

# Drop rows that matched neither condition
cleaned_df_cards = cleaned_df_cards.dropna(subset=["narrative"])
cleaned_df_cards = cleaned_df_cards.dropna(subset=["claim"])

cleaned_df_cards = cleaned_df_cards.drop_duplicates().reset_index(drop=True)

dataset = load_dataset("alextsiak/cc26-cards-aug-300")

df_cc26 = dataset["train"].to_pandas()
df_cc26["narrative"] = df_cc26["narrative"].apply(lambda x: x[0])

merged_df = pd.concat(
    [df[["claim", "narrative"]],
     #cleaned_df_aug[["claim", "narrative"]],
     cleaned_df_cards[["claim", "narrative"]]],
     #df_cc26[["claim", "narrative"]]],
    ignore_index=True
)

merged_df = merged_df.drop_duplicates().reset_index(drop=True)

# Adopt labels according to high levels
high_level_df = merged_df[["claim", "narrative"]].copy()

def simplify_label(s):
    # split on ; first
    parts = s.split(';')

    # extract the first number from each part
    firsts = [re.split(r'[_\-]', p)[0] for p in parts]

    # keep unique while preserving order
    unique = list(dict.fromkeys(firsts))

    return ';'.join(unique)

high_level_df['narrative'] = merged_df['narrative'].apply(simplify_label)

Narratives_list = {
    #"0_0": "No disinformation narrative",
    "1": "Global warming is not happening",
    "2": "Human greenhouse gases are not causing climate change",
    "3": "Climate impacts/global warming is beneficial/not bad",
    "4": "Climate solutions won't work",
    "5": "Climate movement/science is unreliable"
}

Narratives_list_str = "\n".join([f"{k}: {v}" for k, v in Narratives_list.items()])

def build_prompt(claim, Narratives_list_str):
    return f"""You are an expert system for detecting climate change disinformation.

You will be given a single claim. Your task is to classify the claim by assigning the most
appropriate narrative ID(s) from the Narratives_list below.

Instructions:
- Select the narrative ID(s) whose description best matches the claim.
- If the claim clearly aligns with more than one narrative, return all applicable narrative IDs separated by a semicolon (;).
- If the claim does NOT contain climate change disinformation or does not match any listed narrative, return exactly: 0.
- Do NOT explain your reasoning.
- Do NOT output anything other than the narrative ID(s).

Narratives_list:
{Narratives_list_str}

Examples:

Example 1
Claim: "Over the past century, the Earth's temperature has risen by about 0.1°C due to CO2."
Narrative IDs: 1;3

Example 2
Claim: "Interesting fact:  Way back when, CO2 levels were high, but the sun wasn't as strong. 🤔 Makes you wonder about the link between solar activity and climate, right?"
Narrative IDs: 2

Example 3
Claim: "Turns out, species that can adapt easily to different environments are often the ones that can survive in a wide variety of places."
Narrative IDs: 0

Now classify the following claim:

Claim: "{claim}"
Narrative IDs:"""

claims = high_level_df["claim"].tolist()
narratives = high_level_df["narrative"].tolist()

messages_all = []

for i in range(len(claims)):
    #print(i)
    claim = claims[i]
    codes = narratives[i]

    prompt = build_prompt(claim, Narratives_list_str)

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

messages_data = messages_all #load_dataset("json", data_files="cc_messages.jsonl", split = "train")
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
    max_seq_length = max_seq_length,
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

#print(trainer.train_dataset[5]["labels"])

trainer_stats = trainer.train()

def predict_claims(claims, max_new_tokens=16):
    results = []
    model.eval()

    for claim in claims:
        prompt = build_prompt(claim, Narratives_list_str)

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
    pattern = r"Narrative IDs:\n\nassistant\n<think>\n\n</think>\n\n(.+)$"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else None

train_predictions = predict_claims(high_level_df["claim"].tolist(), )
train_extracted_predictions = [
    extract_narrative_ids(pred) for pred in train_predictions
]

df_high_train_pred = high_level_df[["claim", "narrative"]].copy()
df_high_train_pred["narrative"] = train_extracted_predictions
df_high_train_pred.to_csv("Train_High_predictions.csv", index=False)

test_predictions = predict_claims(df_test["claim"].tolist())
test_extracted_predictions = [
    extract_narrative_ids(pred) for pred in test_predictions
]

test_extracted_predictions[53]

df_high_tets_pred = df_test[["claim", "narrative"]].copy()
df_high_test_pred["narrative"] = test_extracted_predictions
df_high_test_pred.to_csv("Test_High_predictions.csv", index=False)

df_high_train_pred = pd.read_csv('/high-level-narratives-predictions/Train_High_predictions.csv')
df_high_test_pred = pd.read_csv('/high-level-narratives-predictions/Test_High_predictions.csv')

Narratives_list = {
    #"0_0": "No disinformation narrative",
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

def build_prompt(claim, high_level_label, Narratives_list_str):
    return f"""You are an expert system for detecting climate change disinformation.

You will be given:
1) A single claim.
2) One or more predicted high_level label(s) (e.g., "2" or "1;3").

Your task is to assign the most appropriate sub_level narrative ID(s) from the Narratives_list **restricted to the given high_level label(s)**.

Instructions:
- Only select sub_level narrative IDs that fall under the provided high_level label(s).
- Choose the narrative ID(s) whose descriptions best match the claim.
- If the claim clearly aligns with more than one narrative, return all applicable narrative IDs separated by a semicolon (;).
- - If none of the sub_level narratives under the given high_level label(s) apply, output the corresponding **generic sub_level label(s)** in the form "X_0" (e.g., "1_0", "3_0"), where X is each provided high_level label.

- Do NOT explain your reasoning.
- Do NOT output anything other than the narrative ID(s).

Narratives_list:
{Narratives_list_str}

Examples:

Example 1
Claim: "According to a recent study, there's no link between CO2 emissions and temperature changes in our lifetime. #ClimateChange #Science"
High_level_label: "2"
Sub_level Narrative IDs: 2_3

Example 2
Claim: "According to recent research, we haven't seen any glacier melting or sea level rise since 1995."
High_level_label: "1"
Sub_level Narrative IDs: 1_1;1_6

Example 3
Claim: "Over the past century, the Earth's temperature has risen by about 0.1°C due to CO2."
High_level_label: "1;3"
Sub_level Narrative IDs: 1_4;3_4

Now classify the following claim:

Claim: "{claim}"
High_level_label: "{high_level_label}"
Sub_level Narrative IDs:"""

claims = df["claim"].tolist()
narratives = df["narrative"].tolist()
high_preds = df_high_train_pred["narrative"].tolist()

messages_all = []

for i in range(len(claims)):
    #print(i)
    claim = claims[i]
    high_level_label = high_preds[i]
    codes = narratives[i]

    prompt = build_prompt(claim, high_level_label, Narratives_list_str)

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

messages_data = messages_all #load_dataset("json", data_files="cc_messages.jsonl", split = "train")
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
    max_seq_length = max_seq_length,
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

#print(trainer.train_dataset[5]["labels"])

trainer_stats = trainer.train()

def predict_claims(claims, high_level_label, max_new_tokens=16):
    results = []
    model.eval()

    for i in range(len(claims)):
        prompt = build_prompt(claims[i], high_level_label[i], Narratives_list_str)

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

test_predictions = predict_claims(df_test["claim"].tolist(), df_high_test_pred['narrative'].tolist())
test_extracted_predictions = [
    extract_narrative_ids(pred) for pred in test_predictions
]

df_submission = df_test[["claim_id", "abstract_id", "narrative"]].copy()
df_submission["narrative"] = test_extracted_predictions

df_submission.to_csv("predictions.csv", index=False)