from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from multiprocessing import cpu_count
import utils
#from qdrant_client import QdrantClient
import numpy as np
import datasets
from datasets import DatasetDict
import torch
from trl import SFTTrainer
from peft import LoraConfig

# Taking Mistral-7B base model for fine-tuning on German news articles
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# prepare tokenizer
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'
tokenizer.model_max_length = 2048

# Load huggingface dataset directly
dataset = datasets.load_dataset("mlsum", "de")

# subset data for dev
dev_size = 7000
indices_train = range(dev_size)
indices_test = range(round(dev_size/4))

dataset_dict = {"train": dataset["train"].shuffle(seed=42).select(indices_train),
                "eval": dataset["validation"].shuffle(seed=42).select(indices_test)}

raw_datasets = DatasetDict(dataset_dict)

column_names = list(raw_datasets["train"].features)

# add feature inst_text to the train and test dataset
raw_datasets = raw_datasets.map(
  utils.add_response_structure,
  fn_kwargs={"mistral_eos_token": tokenizer.eos_token},
  num_proc=cpu_count(), 
  remove_columns=column_names,
  desc="Adding inst_text feature (and remove existing columns)")

# drop rows with very long texts (to avoid indexing issues when longer than max_sequence_length)
raw_datasets = raw_datasets.filter(lambda example: len(tokenizer(example["inst_text"])["input_ids"]) < tokenizer.model_max_length)
print(raw_datasets)

# create the splits
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["eval"]

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model_kwargs = dict(
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    use_cache=False, # gradient checkpointing instead
    device_map=device_map,
    quantization_config=quantization_config,
)

# output path for training checkpoints and logs
output_dir = 'llm_inference/models/'

# training config
training_args = TrainingArguments(
    bf16=True,
    tf32=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=1.2e-04, # up to 2e-04 used for QLoRA
    optim="adamw_torch_fused",  
    log_level="info",
    logging_steps=10,
    logging_strategy="steps",
    lr_scheduler_type="constant", # found to perform best in experiments in QLora-Paper
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    num_train_epochs=2,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1, 
    save_strategy="epoch", 
    save_total_limit=None,
    seed=42,
)

peft_config = LoraConfig(
        r=32, # lower r to to have a higher ratio of fine-tuning tokens to trainable parameters
        lora_alpha=16, # ratio alpha=r/2 commonly used in public experiments
        lora_dropout=0.07, # values between 0.05 and 0.1 common
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
)

# Supervised Fine-Tuning (based on title of articles atm, optimally labels should rather be a substantivierte Themen-Beschreibung, tbd)
trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="inst_text",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )

# continue fine-tuning based on checkpoint (since training data is subsetted atm, ideally data should be offset accordingly, Todo)
train_result = trainer.train(
  #resume_from_checkpoint="llm_inference/models/checkpoint_week_1"
)

#output metrics/state
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()