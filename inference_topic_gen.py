from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import utils

output_dir = "models/topic_generator/checkpoint-6"

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir, load_in_4bit=True, device_map="auto")

# prepare tokenizer
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'
tokenizer.model_max_length = 2048

# example text for dev, connect with frontend and/or vectorDB later for variable text input
# also use utils function to prepare the message later
example_text = "Die Bundesregierung hat die Impfpflicht für Pflegekräfte beschlossen. Die Maßnahme soll dazu beitragen, die Verbreitung des Coronavirus in Pflegeheimen zu verhindern. Die Impfpflicht gilt für alle Pflegekräfte, die in Pflegeheimen arbeiten. Sie müssen sich bis zum 1. März impfen lassen. Wer sich nicht impfen lässt, riskiert seinen Job. Die Bundesregierung will so die Sicherheit der Bewohner in Pflegeheimen erhöhen. Die Impfpflicht ist umstritten. Einige Pflegekräfte sind dagegen. Sie sagen, dass die Impfung gefährlich sein könnte. Andere Pflegekräfte sind dafür. Sie sagen, dass die Impfung wichtig ist, um die Bewohner zu schützen."
example_message = "<system>\n You are a helpful assistent focussing on providing brief and precise answers. Generally answer in German." + tokenizer.eos_token + "\n<user>\n Briefly describe the topic in German based on the following user text: " + example_text + tokenizer.eos_token + "\n<assistent>\n"

# prepare the messages for the model
encoded_input = tokenizer(example_message, truncation=True, return_tensors="pt").to("cuda")
input_ids = encoded_input["input_ids"]

# inference
outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=70,
        do_sample=True,
        temperature=0.7,
        #top_k=50,
        top_p=0.95,
        repetition_penalty=1.03,
)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])