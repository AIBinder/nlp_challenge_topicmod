from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from huggingface_hub import HfFolder

app = Flask(__name__)

# Load model and prepare tokenizer
# if token is provided in .env file, official repo for Mistral is used, otherwise alternative Repo without Gated Access
load_dotenv('.env')
token = os.getenv('HF_READ_TOKEN')
if token:
    model_id = "mistralai/Mistral-7B-v0.1"
    HfFolder.save_token(token)
else:
    model_id = "MaziyarPanahi/Mistral-7B-v0.1"
print("Using repo: " + model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'
tokenizer.model_max_length = 2048

quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config,
        torch_dtype="auto",
        #attn_implementation="flash_attention_2", 
        #install flash-attention and maybe call dtype explicitly as float16 for faster inference later
        device_map=device_map
        )

# Load Lora Adapter
adapter_path = "models/topic_gen_v1"
model.load_adapter(adapter_path)
model.enable_adapters()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # load text
        input_json = request.get_json()
        input_text = input_json.get("input_text")
        encoded_input = tokenizer(input_text, truncation=True, padding=True, return_tensors="pt")
        input_ids = encoded_input["input_ids"].to(device)

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
                repetition_penalty=1.03, #higher repetition penalty to support shorter topic descriptions without repetition
                # maybe also use length_penalty for that 
        )

        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output_text_proc = re.split(r'(<user>|<system>|<assistent>)', output_text)[6]
        # Todo: write cleaner later, Idea is to cut/stop assistant output on <system>, <user> tags and output assistent response only (without query)

        # Todo: account for uppercasing German topic name in fine-tuning training data structure instead of uncleanly here
        target_pos_upper = 25
        if output_text_proc.startswith("\nDas generelle Thema ist") and len(output_text_proc)>target_pos_upper:
            output_text_proc = output_text_proc[:target_pos_upper] + output_text_proc[target_pos_upper].upper() + output_text_proc[target_pos_upper + 1:]

        return_json = {
            "output": output_text_proc 
            }

        return jsonify(return_json), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=True)


