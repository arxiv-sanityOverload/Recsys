# NLTK 

import os, time, nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from multiprocessing import Pool
import re
from nltk import stem

# Global Variables 
stop_words = stopwords.words('english')
stemmer = stem.PorterStemmer()

# Helper Functions.
num_partitions = 1000 # number of partitions to split dataframe
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool()
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def clean_dataframe(df):
    start = time.time()
    print('Process working on: ',os.getpid())
    df['tokenized'] = df['title'].apply(lambda x: apply_all(x)) + df['abstract'].apply(lambda x: apply_all(x))
    print('Process done:',os.getpid())
    print()
    end = time.time()
    print("time to complete :", end-start)
    return df

# For data cleaning 
def initial_clean(text):
    """
    Function to clean text of websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", " ", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text

def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stop_words]

def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    """
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text

def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(initial_clean(text)))

def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    # lets keep with the p,q notation above
    p = query[None,:].T # take transpose
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

def get_most_similar_documents(query,matrix,k=100):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims.argsort()[:k] # the top k positional index of the smallest Jensen Shannon distances