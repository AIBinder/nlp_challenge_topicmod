# Todo: Funktionen noch verständlicher zu benennen

# system prompt
def query_string(text,mistral_eos_token):
    return "<system>\n You are a helpful assistent focussing on providing the topic of a text briefly and precisely. Generally answer in German." + mistral_eos_token + "\n<user>\n Briefly state the topic of the following user text in German: " + text + mistral_eos_token + "\n<assistent>\n"

# response structure for fine-tuning
def add_feature(example,mistral_eos_token):
    example["inst_text"] = query_string(example["text"],mistral_eos_token) + "Das generelle Thema ist " + example["topic"] + ". \n " + example["title"] + mistral_eos_token
    return example

