import pandas as pd

# Rem: data_prep_mlsum to_be_removed probably/ not required anymore
# (only needed for fine-tuning encoder LLM to classify topic based on text embeddings)
def data_prep_mlsum(dataset):
    train_dataset = dataset["train"]
    #eval_dataset = dataset["validation"]
    #test_dataset = dataset["test"]

    classes = dataset["train"]["topic"]
    class_series = pd.Series(classes)

    # get the distribution of classes
    #print(class_series.value_counts())

    most_common_topics = class_series.value_counts().index[:10].to_list()

    #filter dataset to the 10 most common topics (and exclude very long texts)
    filtered_train_dataset = train_dataset.filter(lambda example: example["topic"] in most_common_topics and len(example["text"]) < 3850)
    #print(len(filtered_train_dataset))

    return filtered_train_dataset

def query_string(text,mistral_eos_token):
    return "<system>\n You are a helpful assistent focussing on providing the topic of a text briefly and precisely. Generally answer in German." + mistral_eos_token + "\n<user>\n Briefly state the topic of the following user text in German: " + text + mistral_eos_token + "\n<assistent>\n"

def add_feature(example,mistral_eos_token):
    example["inst_text"] = query_string(example["text"],mistral_eos_token) + "Das generelle Thema ist " + example["topic"] + ". \n " + example["title"] + mistral_eos_token

    return example

