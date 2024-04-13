from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.de.stop_words import STOP_WORDS
from qdrant_client import QdrantClient
import numpy as np

# BERTopic for encoder based topic modeling
topic_model = BERTopic(
    #embedding_model="aari1995/German_Semantic_STS_V2",
    vectorizer_model=CountVectorizer(stop_words=list(STOP_WORDS), min_df=2), # Rem.: custom token pattern required to account for German Umlaute
    #umap_model=umap_model,
    #hdbscan_model=hdbscan_model,
    #representation_model=representation_model,

    # Hyperparameters
    #top_n_words=10,
    #calculate_probabilities=False, 
    verbose=True)

#get embedding vectors
client = QdrantClient(url="http://localhost:6333")

# retrieve the first 100 vectors from the collection
data = client.retrieve(
    collection_name="articles",
    ids=list(range(100)),
    with_vectors= True,
    with_payload=True
)

embedding_vectors = np.array([d.vector for d in data])
data_snippet = [d.payload["page_content"] for d in data]

# Create topic model
topics, probs = topic_model.fit_transform(data_snippet, embedding_vectors)


print(topic_model.get_topic_info())
print(topic_model.get_topic(1, full=True))



