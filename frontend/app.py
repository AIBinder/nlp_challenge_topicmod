import streamlit as st
import requests
import json
import utils

st.title('Topic Generator')
st.write('Ausgabe des Themas von deutschen News-Artikeln. [Dev-Version v1]')

# Define textual input fields
text = st.text_input("Text", "")
eos_token="</s>" #Todo: invoke from tokenizer later
message = utils.query_string(text,eos_token)
# Placeholder for model output
output_placeholder = st.empty()

# URL of your Flask app's prediction endpoint
url = 'http://llm-inference:5000/predict'

headers = {'Content-Type': 'application/json'}

# Button to trigger model call
if st.button('Submit'):
    # Sample data to send for prediction
    data = {'input_text': message}
    # Send a POST request with input data
    response = requests.post(url, json=data, headers=headers)
    model_output = response.json()
    output_placeholder.write(model_output["output"])