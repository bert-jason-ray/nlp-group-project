import unicodedata
import re
import json
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

import pandas as pd
import acquire
from time import strftime

import warnings
warnings.filterwarnings('ignore')

def basic_clean(string):
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    string = unicodedata.normalize('NFKD', string)\
             .encode('ascii', 'ignore')\
             .decode('utf-8', 'ignore')
    string = re.sub(r'[^\w\s]', '', string).lower()
    #string = re.sub(r'(\&#9;)', '', string)
    #string = re.sub("&#9;","",string)
    #string = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", string)
    #string = re.sub('\W+','', string)
    #string = re.sub(r'\s+', '', string)
    return string

#def basic_clean(text):
    #text = (unicodedata.normalize('NFKD', text.lower())
            #.encode('ascii', 'ignore') # ascii to reduce noise
            #.decode('utf-8', 'ignore') # decode using utf-8
           #)
    #return re.sub(r"[^a-z0-9\s]", '', text)


def tokenize(string):
    '''
    This function takes in a string and
    returns a tokenized string.
    '''
    # Create tokenizer.
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    # Use tokenizer
    string = tokenizer.tokenize(string, return_str = True)
    
    return string

def stem(string):
    '''
    This function takes in a string and
    returns a string with words stemmed.
    '''
    # Create porter stemmer.
    ps = nltk.porter.PorterStemmer()
    
    # Use the stemmer to stem each word in the list of words we created by using split.
    stems = [ps.stem(word) for word in string.split()]
    
    # Join our lists of words into a string again and assign to a variable.
    string = ' '.join(stems)
    
    return string

def lemmatize(string):
    '''
    This function takes in string for and
    returns a string with words lemmatized.
    '''
    # Create the lemmatizer.
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Use the lemmatizer on each word in the list of words we created by using split.
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    # Join our list of words into a string again and assign to a variable.
    string = ' '.join(lemmas)
    
    return string

def remove_stopwords(string, extra_words = [], exclude_words = []):
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    # Create stopword_list.
    stopword_list = stopwords.words('english')
    
    # Remove 'exclude_words' from stopword_list to keep these in my text.
    stopword_list = set(stopword_list) - set(exclude_words)
    
    # Add in 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))

    # Split words in string.
    words = string.split()
    
    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [word for word in words if word not in stopword_list]
    
    # Join words in the list back into strings and assign to a variable.
    string_without_stopwords = ' '.join(filtered_words)
    
    return string_without_stopwords

def drop_data(df):
    '''
    This function takes in the repo dataframe
    Drops any rows with nulls
    '''
    df = df.dropna()
    #df['language'] = df['language'].value_counts().loc[lambda x : x>5]
    #df = df.reset_index()
    #df = df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df = df.reindex(index=my_index)
    return df

def get_top_languages(df):
    '''
    This function takes in a dataframe and returns the top four
    programming languages found in the data
    '''
    top_6_list = list(df.language.value_counts().head(6).index)
    mask = df.language.apply(lambda x: x in top_6_list)
    df = df[mask]
    return df


def prep_github_data(df, column = 'readme_contents', extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the content (in string) for the column 
    with an option to pass lists for additional stopwords (extra_words)
    and an option to pass words to exclude from stopwords (exclude words)
    returns a df with the  original text, cleaned (tokenized and stopwords removed),
    stemmed text, lemmatized text.
    '''
    
    df = drop_data(df)

    df = get_top_languages(df)

    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, exclude_words=exclude_words)
    df['clean'] = df['clean'].str.replace("&#9;","")

    df['stemmed'] = df['clean'].apply(stem)
    
    df['lemmatized'] = df['clean'].apply(lemmatize)

    

    return df

def split_github_data(df):
    '''
    This function performs split on github data, stratify language.
    Returns train, validate, and test dfs.
    '''
    train, test = train_test_split(df, test_size=.2, 
                                        random_state=123, stratify=df.language)
    #train, validate = train_test_split(train_validate, test_size=.3, 
                                   #random_state=123, stratify=train_validate.language)

    print('train--->', train.shape)
    #print('validate--->', validate.shape)
    print('test--->', test.shape)
    return train, test

