import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic

import torch
torch.cuda.empty_cache()

itr = 7
min_cluster_size = 350
save_path = "/mnt/scratch/lellaom/narrative_theme_identification/results/itr"+str(itr)+"_"+str(min_cluster_size)+"/"

def preprocess_and_split_doc(doc):
    doc = re.sub(r'\s+', ' ', doc.strip())
    words = doc.split()

    docs = []
    for i in range(0, len(words), 250):
        docs.append(" ".join(words[i:(i+250)]))

    return docs

booknames = []
docs = []
directory = '/mnt/scratch/lellaom/narrative_theme_identification/data/Gutenberg/txt'
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            doc = file.read()
            temp = preprocess_and_split_doc(doc)
            booknames.extend([filename] * len(temp))
            docs.extend(temp)

df = {"Book Name": booknames}
df = pd.DataFrame(df)

custom_stop_words = []
stop_words = list(CountVectorizer(stop_words="english").get_stop_words())
stop_words.extend(custom_stop_words)

vectorizer_model = CountVectorizer(stop_words=stop_words)
ctfidf_model = ClassTfidfTransformer()
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', low_memory=False)
hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
representation_model = KeyBERTInspired()

model = BERTopic(vectorizer_model=vectorizer_model,
                 ctfidf_model=ctfidf_model,
                 representation_model=representation_model,                 
                 min_topic_size=min_cluster_size,
                 nr_topics=None,
                 low_memory=False,
                 umap_model=umap_model,
                 hdbscan_model=hdbscan_model,
                 calculate_probabilities=False,
                 verbose=True)
topics, probs = model.fit_transform(docs)
new_topics = model.reduce_outliers(docs, model.topics_, strategy="embeddings")
documents = pd.DataFrame({"Document": docs, "Topic": new_topics})
model._update_topic_size(documents)
model.save(save_path+"model", serialization="safetensors", save_ctfidf=True)

joblib.dump(model.get_topic_info(), save_path+"topic_info_df.joblib")
joblib.dump(model.get_document_info(docs, df=df), save_path+"doc_info_df.joblib")