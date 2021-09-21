import re
import pandas
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacytextblob')


def subcatego(cat_mess: str) -> str:
    r = re.compile('slug...([a-zA-Z 0-9]+)/([a-zA-Z 0-9]+)')
    f = r.findall(cat_mess)
    if len(f)>0:
        return f[-1]  # only difference here
    else:
        s = re.compile('slug...([a-zA-Z 0-9]+)')
        g = s.findall(cat_mess)
        return g[0]


def catego(cat_mess: str) -> str:
    r = re.compile('slug...([a-zA-Z 0-9]+)/([a-zA-Z 0-9]+)')
    f = r.findall(cat_mess)
    if len(f)>0:
        return f[0]  # only difference here
    else:
        s = re.compile('slug...([a-zA-Z 0-9]+)')
        g = s.findall(cat_mess)
        return g[0]


def wrangle2(df):
    """
    For cleaning pd.DataFrame read from csv available at
    <https://webrobots.io/kickstarter-datasets/>
    """
    df = df.copy()
    
    observation_threshold = len(df)/2
    df.dropna(thresh=observation_threshold , axis=1, inplace=True)
    df.dropna(subset=['blurb'], inplace=True)
    df.dropna(subset=['location'], inplace=True)
    df.drop_duplicates('id',inplace=True)
    # df.set_index('id',inplace=True)
    df.reset_index(drop=True,inplace=True)

    # class_to_drop1 = df[df['state'] == 'canceled'].index
    # df.drop(class_to_drop1, inplace=True)
    df = df.loc[df.state != 'live']
    df.loc[df.state == 'canceled', 'state'] = 0
    df.loc[df.state == 'failed', 'state'] = 0
    df.loc[df.state == 'successful'] = 1
    df = df.rename(columns={'state': 'success'})

    df['goal_usd'] = round(df['goal'] * df['static_usd_rate'],2)

    for col in ['created_at', 'deadline', 'launched_at']:
        df[col] = pandas.to_datetime(df[col], origin='unix', unit='s')
    df['launch_day'] = df['launched_at'].dt.dayofweek
    days = df['deadline'] - df['launched_at']
    df['campaign_duration_in_days'] = days.dt.round('d').dt.days

    # df['subcategory'] = df['category'].apply(lambda x: subcatego(x)) #def above
    # df['category'] = df['category'].apply(lambda x: catego(x))       #def above

    df['name_blurb'] = [i for i in zip(df.name, df.blurb)]
    
    droppable=['converted_pledged_amount','creator','currency','currency_symbol','currency_trailing_code','current_currency','fx_rate','photo','pledged','profile','slug','source_url','state_changed_at','urls', 'usd_type','spotlight','staff_pick','usd_pledged','backers_count', 'is_starrable','disable_communication','goal','static_usd_rate','location','country_displayable_name','name','blurb']
    df = df.drop(droppable, axis=1)
    
    # df = df[df.columns[[4,6,7,9,0,10,2,5,8,3,11]]]

    return df


def blurb_quant(name_blurb):

    """
    Accepts a tuple (kickstarter name, kickstarter blurb) and
    returns a dataframe of numeric features *possibly* relevant to ks success.
    
    Requires re, pandas, nltk.sentiment.vader, spacy, spacytextblob:
    --> sid = vader.SentimentIntensityAnalyzer()
    --> nlp = spacy.load('en_core_web_lg')
    --> nlp.add_pipe('spacytextblob')

    Usage: blurb_stats = df['name_blurb'].apply(lambda x: blurb_quant(x))
    """

    name = str(name_blurb[0])
    blurb = str(name_blurb[1])
    nm_blb = name+': '+blurb

    # basic counts
    name_chars = len(name)
    name_words = len(name.split())
    blb_chars = len(blurb)
    blb_words = len(blurb.split())
    name_in_blb = int(name.lower() in blurb.lower())
    basic_stats = pandas.Series(
        data = [name_chars, name_words, blb_chars, blb_words, name_in_blb],
        index = ['name_chars', 'name_words', 'blb_chars', 'blb_words','name_in_blb'])

    # token tallies
    doc2 = nlp(nm_blb)
    stop = [token.is_stop for token in doc2]
    oov = [token.is_oov for token in doc2]
    pron = [token.pos_ for token in doc2 if token.pos_=='PRON']
    adj = [token.pos_ for token in doc2 if token.pos_=='ADJ']
    noun = [token.pos_ for token in doc2 if token.pos_=='NOUN']
    shout = [1 for token in doc2 if re.findall('X{4,}', token.shape_)]
    caps = [1 for token in doc2 if re.findall('Xx+', token.shape_)]
    # [ent.label_ for ent in doc2.ents]
    token_stats = pandas.Series(
        data = [sum(stop), sum(oov), len(pron), len(adj), len(noun), len(shout), len(caps)],
        index = ['stopwords', 'nonwords', 'pron', 'adj', 'noun', 'shout', 'capwords'])

    # blurb sentiment
    doc = nlp(blurb)                     # spacy & textblob
    norm = doc.vector_norm
    subj = doc._.subjectivity
    pol = doc._.polarity
    vader = sid.polarity_scores(blurb)   # nltk & vader
    pos = vader['pos']
    neg = vader['neg']
    neu = vader['neu']
    comp = vader['compound']
    blurb_sentiment = pandas.Series(
        data = [norm, subj, pol, pos, neg, neu, comp],
        index = ['norm', 'subj', 'pol', 'pos', 'neg', 'neu', 'comp'])

    blurb_stats = pandas.concat([basic_stats, token_stats, blurb_sentiment])

    return blurb_stats

