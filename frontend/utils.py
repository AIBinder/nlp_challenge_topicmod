def query_string(text,mistral_eos_token):
    return "<system>\n You are a helpful assistent focussing on providing the topic of a text briefly and precisely. Generally answer in German." + mistral_eos_token + "\n<user>\n Briefly state the topic of the following user text in German: " + text + mistral_eos_token + "\n<assistent>\n"

