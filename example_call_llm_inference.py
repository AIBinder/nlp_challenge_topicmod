import requests
import utils
import json

# Rem.: localhost port for test here, in frontend container the inference-API can then be called internally via docker-network
url = 'http://localhost:5007/predict'
eos_token="</s>" #invoke from tokenizer then

example_text = "Die Bundesregierung hat die Impfpflicht für Pflegekräfte beschlossen. Die Maßnahme soll dazu beitragen, die Verbreitung des Coronavirus in Pflegeheimen zu verhindern. Die Impfpflicht gilt für alle Pflegekräfte, die in Pflegeheimen arbeiten. Sie müssen sich bis zum 1. März impfen lassen. Wer sich nicht impfen lässt, riskiert seinen Job. Die Bundesregierung will so die Sicherheit der Bewohner in Pflegeheimen erhöhen. Die Impfpflicht ist umstritten. Einige Pflegekräfte sind dagegen. Sie sagen, dass die Impfung gefährlich sein könnte. Andere Pflegekräfte sind dafür. Sie sagen, dass die Impfung wichtig ist, um die Bewohner zu schützen."
example_message = utils.query_string(example_text,eos_token)

input_data = {'input_text':example_message}

headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=input_data, headers=headers)

print(response.status_code)
print(response.json())