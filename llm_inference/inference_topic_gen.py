from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model
model_id = "mistralai/Mistral-7B-v0.1"
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
        torch_dtype="auto", #maybe call explicitly as float16 later for performance (analogously during fine-tuning)
        #attn_implementation="flash_attention_2", #integrate cuda base image for faster inference later
        device_map=device_map
        )

# Load Lora Adapter
adapter_path = "models/checkpoint-46"
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
                repetition_penalty=1.03,
        )

        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # manual stopping criterion on <user> in string
        output_text_processed = re.split(r'(<user>|<system>|<assistent>)', output_text)[6]
        # Todo: write cleaner later, Idea is to cut assistant output on <system>, <user> tags and output assistent response only (without query)

        print(output_text_processed)
        return_json = {
            "output": output_text_processed 
            }

        return jsonify(return_json), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=True)


