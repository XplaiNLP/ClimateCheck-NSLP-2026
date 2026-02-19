import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import torch
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only

dataset_cc = load_dataset('alextsiak/cc26-cards-aug-300', split='train')


TAXONOMY = {
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

taxonomy_str = "\n".join([f"{k}: {v}" for k, v in TAXONOMY.items()])


def build_prompt(claim):
    return f"""You are an expert in detecting climate change related disinformation. 
    You will be given a single claim. Your task is to classify the claim by assigning the most appropriate narrative ID(s) from the taxonomy below. 

Instructions:

    - Select the narrative ID(s) whose description best matches the claim.
    - If the claim clearly aligns with more than one narrative, return all applicable narrative IDs separated by a semicolon (;).
    - If the claim does NOT contain climate change disinformation or does not match any listed narrative, return exactly: 0_0.
    - Do NOT explain your reasoning.
    - Do NOT output anything other than the narrative ID(s).

Narrative IDs:
{taxonomy_str}

Now, classify the following claim:

Claim: "{claim}"
Narrative IDs:""" 

df = dataset_cc.to_pandas()
claims = df["claim"].tolist()
narratives = df["narrative"]

messages_all = []
import json
for i in range(len(claims)):
    claim = claims[i]
    codes = narratives[i]
    codes_str = ";".join(codes)
    
    prompt = build_prompt(claim)

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": codes_str}
    ]

    messages_all.append({"messages": messages})


with open("cc_messages.jsonl", 'w', encoding='utf-8') as f:
    for message in messages_all:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


max_seq_length = 8192
load_in_4bit = False
dtype = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
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
    use_gradient_checkpointing = False, 
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

dataset = load_dataset("json", data_files="cc_messages.jsonl", split = "train")

tokenizer = get_chat_template(
    tokenizer,
#    chat_template = "qwen-2.5", 
    #chat_template = "qwen-3",
    chat_template = 'llama-3'

)

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            messages, 
            #tokenize=False,
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
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 2,
        #max_steps=50,
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
    instruction_part = "<|start_header_id|>user<|end_header_id|>",
    response_part    = "<|start_header_id|>assistant<|end_header_id|>",
)

print(trainer.train_dataset[5]["labels"])

trainer_stats = trainer.train()

model.save_pretrained("llama8b")
tokenizer.save_pretrained("llama8b")