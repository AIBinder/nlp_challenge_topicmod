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

# Taking Mistral base model for fine-tuning on German texts (due to good performance generally)
model_id = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# prepare tokenizer
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'
tokenizer.model_max_length = 2048

""" #Optional: get data from vectorDB
client = QdrantClient(url="http://localhost:6333")

# retrieve the first 1000 vectors from the collection
data = client.retrieve(
    collection_name="articles",
    ids=list(range(1000)),
    with_vectors= True,
    with_payload=True
)

client.close()
embedding_vectors = np.array([d.vector for d in data])
# Rem.: maybe prepare inst_text as metadata in vectorDB directly and only load here accordingly """

# Load huggingface dataset directly
dataset = datasets.load_dataset("mlsum", "de")

# subset data for dev
dev_size = 2000
indices_train = range(0,dev_size)
indices_test = range(0,round(dev_size/4))

dataset_dict = {"train": dataset["train"].select(indices_train),
                "test": dataset["test"].select(indices_test)}

raw_datasets = DatasetDict(dataset_dict)

column_names = list(raw_datasets["train"].features)

# add feature inst_text to the train and test dataset
raw_datasets = raw_datasets.map(
  utils.add_feature,
  fn_kwargs={"mistral_eos_token": tokenizer.eos_token},
  num_proc=cpu_count(), 
  remove_columns=column_names,
  desc="Adding inst_text feature (and remove existing columns)")

# drop rows with very long texts (to avoid indexing issues when longer than max_sequence_length)
raw_datasets = raw_datasets.filter(lambda example: len(example["inst_text"]) < 5000)

print(raw_datasets)

# create the splits
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]


# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
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
output_dir = 'models/topic_generator'

# training config
training_args = TrainingArguments(
    bf16=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=128,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1, 
    save_strategy="epoch", 
    save_total_limit=None,
    seed=42,
)

# based on config
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

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


train_result = trainer.train()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()