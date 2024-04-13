import pandas as pd

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