import pandas as pd
from bertopic import BERTopic
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

itr = 8
min_cluster_size = 400
results_path = "/mnt/scratch/lellaom/narrative_theme_identification/results/itr"+str(itr)+"_"+str(min_cluster_size)+"/"

df = pd.read_csv(results_path+"doc_info.csv")
docs = df["Document"].tolist()
topic_model = BERTopic.load(results_path+"model/")
topics = topic_model.topics_

documents = pd.DataFrame({"Document": docs,
                          "ID": range(len(docs)),
                          "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

vectorizer = topic_model.vectorizer_model
analyzer = vectorizer.build_analyzer()

tokens = [analyzer(doc) for doc in cleaned_docs]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]

topic_words = []
for topic in range(0, 25):
    words = list(zip(*topic_model.get_topic(topic)))[0]
    words = [word for word in words if word in dictionary.token2id]
    topic_words.append(words)
topic_words = [words for words in topic_words if len(words) > 0]
print(len(topic_words))

coherence_model = CoherenceModel(topics=topic_words, 
                                 texts=tokens, 
                                 corpus=corpus,
                                 dictionary=dictionary, 
                                 processes=4,
                                 coherence='c_v')
coherence = coherence_model.get_coherence()
print(coherence)