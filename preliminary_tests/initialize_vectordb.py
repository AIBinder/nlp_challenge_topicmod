from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
import datasets
import pandas as pd
import utils

# Load the dataset
dataset = datasets.load_dataset("mlsum", "de")
# preprocess and filter the dataset
filtered_train_dataset = utils.data_prep_mlsum(dataset)

# temp subset for dev
data_snippet = filtered_train_dataset["text"][0:100]

# use German specific embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="aari1995/German_Semantic_STS_V2"
)

# Initialize vectorDB, distance strategy already defaults to Cosine via langchain integration
# REM: use docker network instead of localhost port later
doc_store = Qdrant.from_texts(
    data_snippet, 
    embeddings, 
    url="http://localhost:6333", 
    collection_name="articles",
    ids=list(range(len(data_snippet))),
    metadatas=[{"category": filtered_train_dataset["topic"][i]} for i in range(len(data_snippet))],
    force_recreate=True
)