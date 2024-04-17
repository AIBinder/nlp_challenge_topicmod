import streamlit as st
import requests
import json
import utils

st.title('Topic Generator')
st.write('Ausgabe des Themas zu deutschen Nachrichten-Artikeln (aktuell in der ersten Dev-Version). Hier können entsprechend Artikel in Textform übergeben werden.')

# Define textual input field
text = st.text_input("Text", "")
eos_token="</s>" #Todo: invoke from tokenizer later
# prepare text in format that was used for fine-tuning
message = utils.query_string(text,eos_token)
# Placeholder for model output
output_placeholder = st.empty()

# URL of Flask LLM-inference endpoint
url = 'http://llm-inference:5000/predict'

headers = {'Content-Type': 'application/json'}

# Button to trigger model call
if st.button('Submit'):
    # POST request with prepared text as json data
    data = {'input_text': message}
    response = requests.post(url, json=data, headers=headers)
    model_output = response.json()
    output_placeholder.write(model_output["output"])