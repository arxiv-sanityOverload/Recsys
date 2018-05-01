import pandas as pd
from utility import *
import pymysql.cursors
from nltk import FreqDist
#LDA 
import gensim
from gensim.models import LdaMulticore
from gensim import models, corpora, similarities

# Database connection 
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='password123',
                             db='arxivOverload',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
cursor = connection.cursor()
print("Connected with Database")
query = "SELECT arxiv_id, title, abstract FROM arxivOverload.METADATA"
metadata_table_dataframe = pd.read_sql(query, con=connection)
metadata_table_dataframe = metadata_table_dataframe.drop_duplicates()
data = parallelize_dataframe(metadata_table_dataframe, clean_dataframe)

# drop for saving memory
data = data.drop(['title', 'abstract'], axis=1)

# first get a list of all words
all_words = [word for item in list(data['tokenized']) for word in item]
# use nltk fdist to get a frequency distribution of all words
fdist = FreqDist(all_words)
k = 50000
top_k_words = fdist.most_common(k)

# trimming tokens
top_k_words,_ = zip(*fdist.most_common(k))
top_k_words = set(top_k_words)
def keep_top_k_words(text):
    return [word for word in text if word in top_k_words]

def train_lda(data):
    """
    This function trains the lda model
    We setup parameters like number of topics, the chunksize to use in Hoffman method
    We also do 10 passes of the data since this is a small dataset, so we want the distributions to stabilize
    """
    num_topics = 1000
    chunksize = 30000
    dictionary = corpora.Dictionary(data['tokenized'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
    t1 = time.time()
    # low alpha means each document is only represented by a small number of topics, and vice versa
    # low eta means each topic is only represented by a small number of words, and vice versa
    lda = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=10, iterations=50 ,workers = 3)
    t2 = time.time()
    print("Time to train LDA model on ", len(data), "articles: ", (t2-t1)/60, "min")
    return dictionary,corpus,lda

dictionary,corpus,lda = train_lda(data)

# Figure out to multiprocess calculating similarity

doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])

def get_similarity(data, doc_topic_dist):
    for i in range(len(data)):
        bow = dictionary.doc2bow(data.iloc[10, 3])
        doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=bow)])
        most_sim_ids = get_most_similar_documents(doc_distribution,doc_topic_dist)
        most_similar_df = data[data.index.isin(most_sim_ids)]
        

