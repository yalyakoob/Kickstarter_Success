import re
from itertools import combinations # for feature engineering interactions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy import displacy
# from sklearn.feature_extraction.text import CountVectorizer    # for LDA


nlp = spacy.load("en_core_web_lg")
# df = pd.read_csv('./MP_datasets/Kickstarter20_21_Consolidated.csv')
nlp.add_pipe('spacytextblob')



df['blb_char_count'] = df.blurb.apply(lambda x: len(x))
df['blb_word_count'] = df.blurb.apply(lambda x: len(x.split()))
df['blb_word_len'] = df['blb_char_count']/df['blb_word_count']

def blurb_nums(blurb):

    """Accepts a blurb, returns features that might be relevant to funding.
    Requires nltk.sentiment.vader, spacy, spacytextblob:
    
    --> sid = SentimentIntensityAnalyzer()
    --> nlp = spacy.load('en_core_web_lg')
    --> nlp.add_pipe('spacytextblob')
    """

    blurb = str(blurb)

    vader = sid.polarity_scores(blurb)
    pos = vader['pos']
    neg = vader['neg']
    neu = vader['neu']
    comp = vader['compound']

    doc = nlp(blurb)
    norm = doc.vector_norm
    subj = doc._.subjectivity
    pol = doc._.polarity
    
    blb_stats = pos, neg, neu, comp, norm, subj, pol

    return blb_stats

df['blurb_stats'] = df.blurb.apply(lambda x: blurb_nums(x))



