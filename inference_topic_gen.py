from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import utils
import re

output_dir = "models/topic_generator/checkpoint-23"

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir, load_in_4bit=True, device_map="auto")


# example text for dev, connect with frontend and/or vectorDB later for variable text input
# also use utils function to prepare the message later
example_text = "Die Bundesregierung hat die Impfpflicht für Pflegekräfte beschlossen. Die Maßnahme soll dazu beitragen, die Verbreitung des Coronavirus in Pflegeheimen zu verhindern. Die Impfpflicht gilt für alle Pflegekräfte, die in Pflegeheimen arbeiten. Sie müssen sich bis zum 1. März impfen lassen. Wer sich nicht impfen lässt, riskiert seinen Job. Die Bundesregierung will so die Sicherheit der Bewohner in Pflegeheimen erhöhen. Die Impfpflicht ist umstritten. Einige Pflegekräfte sind dagegen. Sie sagen, dass die Impfung gefährlich sein könnte. Andere Pflegekräfte sind dafür. Sie sagen, dass die Impfung wichtig ist, um die Bewohner zu schützen."
example_message = utils.query_string(example_text,tokenizer.eos_token)

# prepare the messages for the model
encoded_input = tokenizer(example_message, truncation=True, padding=True, return_tensors="pt").to("cuda")
input_ids = encoded_input["input_ids"]

# set attention mask and padding token id
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
attention_mask = input_ids.ne(tokenizer.pad_token_id).int()

# inference
outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=70,
        do_sample=True,
        temperature=0.75,
        #top_k=50,
        top_p=0.95,
        repetition_penalty=1.03,
)

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
# manual stopping criterion on <user> in string
output_text_processed = re.split(r'(<user>|<system>|<assistent>)', output_text)[6]
# Todo: write cleaner later, Idea is to cut assistant output on <system>, <user> tags and output assistent response only (without query)

print(output_text_processed)