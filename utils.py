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

def add_feature(example,mistral_eos_token):
    example["inst_text"] = "<system>\n You are a helpful assistent focussing on providing brief and precise answers. Generally answer in German." + mistral_eos_token + "\n<user>\n Briefly describe the topic in German based on the following user text: " + example["text"] + mistral_eos_token + "\n<assistent>\n" + "Das Thema ist " + example["topic"] + ".\n Zusammenfassung: " + example["summary"] + mistral_eos_token

    return example