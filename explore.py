"""
This explore.py file is created to allow us to imoport our exploration vizuals to our final notebook for questions and takeaways.
please go to https://github.com/bert-jason-ray/nlp-group-project/blob/main/contreras_explore.ipynb to see the vizuals and takeaways

"""

""" The following imports are placed here as well to allow the data to run correclty on the coding platform"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
import pandas as pd
from env import github_token, github_username
import prepare
import acquire
import unicodedata
import re
import json
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filter="ignore"
from bs4 import BeautifulSoup


""" acquiring the data and renaming the dataframe"""
# acquiring data
df = acquire.get_github_data()
# bringing in prepped data
df = prepare.prep_github_data(df,column = 'readme_contents', extra_words=[], exclude_words=['musicbot'])
# dropping unnecesarry columns 
df = df.drop(columns = ['readme_contents', 'stemmed','clean'])
# droppung Null values
df = df.dropna()
# defining train test and validate using import file 
train, validate, test = prepare.split_github_data(df)
""" Using lemmatized data to create all and specific coding language text files from read me"""
# Set up word counts dataframe
all_text = ' '.join(train.lemmatized)
javascript_text = ' '.join(train[train.language == 'JavaScript'].lemmatized)
python_text = ' '.join(train[train.language == 'Python'].lemmatized)
typescript_text = ' '.join(train[train.language == 'TypeScript'].lemmatized)
java_text = ' '.join(train[train.language == 'Java'].lemmatized)
go_text = ' '.join(train[train.language == 'Go'].lemmatized)
kotlin_text = ' '.join(train[train.language == 'Kotlin'].lemmatized)


 
def programing_language_distribution():
    """Starting our exploration we quickly noticed a trend of a handful of programing languages dominating our data.
    As a result, we made the decision to focus on these main languages in an effort build a model more accurately"""

    sns.set_theme(style="white")
    # Code that counts the number of times each vallue is used .
    ax = sns.countplot(x="language", data=train, palette="Set3",order = train['language'].value_counts().index)
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('Programming Language (Readme.md)', size = 16)
    ax.set_ylabel('Count / Frequency', size = 16)
    ax.set_title("Programming Language Distribution", size = 20)
    plt.show()

def all_top_words():
    """ This function allows us to vizualize the top 30 wordsfor all and top 30 words for each coding languages"""
    all_freq = pd.Series(str(all_text).split()).value_counts()
    javascript_freq = pd.Series(str(javascript_text).split()).value_counts()
    python_freq = pd.Series(str(python_text).split()).value_counts()
    typeScript_freq = pd.Series(str(typescript_text).split()).value_counts()
    java_freq = pd.Series(str(java_text).split()).value_counts()
    go_freq = pd.Series(str(go_text).split()).value_counts()
    kotlin_freq = pd.Series(str(kotlin_text).split()).value_counts()
    word_counts = pd.concat([all_freq, javascript_freq, python_freq, typeScript_freq,java_freq, go_freq, kotlin_freq], sort=True, axis=1)
    word_counts.columns = ['all', 'JavaScript', 'python', 'typescript','java', 'go', 'kotlin']
    word_counts = word_counts.fillna(0).apply(lambda s: s.astype(int))
    top_30 = word_counts.sort_values(by='all', ascending=False).head(30)
    return top_30

def word_distribution_vizual():
    """ This function allows us to see the word distribution rates from all, python, javascipt, typescript, java, go, and kotlin coding laguages in a bar chart across all read mes"""
    all_text = ' '.join(train.lemmatized)
    javascript_text = ' '.join(train[train.language == 'JavaScript'].lemmatized)
    python_text = ' '.join(train[train.language == 'Python'].lemmatized)
    typescript_text = ' '.join(train[train.language == 'TypeScript'].lemmatized)
    java_text = ' '.join(train[train.language == 'Java'].lemmatized)
    go_text = ' '.join(train[train.language == 'Go'].lemmatized)
    kotlin_text = ' '.join(train[train.language == 'Kotlin'].lemmatized)
    all_freq = pd.Series(str(all_text).split()).value_counts()
    javascript_freq = pd.Series(str(javascript_text).split()).value_counts()
    python_freq = pd.Series(str(python_text).split()).value_counts()
    typeScript_freq = pd.Series(str(typescript_text).split()).value_counts()
    java_freq = pd.Series(str(java_text).split()).value_counts()
    go_freq = pd.Series(str(go_text).split()).value_counts()
    kotlin_freq = pd.Series(str(kotlin_text).split()).value_counts()
    word_counts = pd.concat([all_freq, javascript_freq, python_freq, typeScript_freq,java_freq, go_freq, kotlin_freq], sort=True, axis=1)
    word_counts.columns = ['all', 'JavaScript', 'python', 'typescript','java', 'go', 'kotlin']
    word_counts = word_counts.fillna(0).apply(lambda s: s.astype(int))
    # Visualize word distribution
    word_counts.sort_values(by='all', ascending=False)[['all','JavaScript', 'python', 'typescript','java', 'go', 'kotlin']].head(9).plot.bar()
    plt.title('Words Used Most in README Files')
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('Nine Most Common Words Used')
    plt.ylabel('Word Quantity')
    plt.show()

def all_words_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    all_D = pd.Series(str(all_text).split())
    top_20_all_words_bigrams = (pd.Series(nltk.ngrams(all_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_all_words_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used on all languages')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def all_words_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    all_D = pd.Series(str(all_text).split())
    top_20_all_words_bigrams = (pd.Series(nltk.ngrams(all_D, 2))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_all_words_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used on all languages')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def all_words_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    all_D = pd.Series(str(all_text).split())
    top_20_all_words_trigrams = (pd.Series(nltk.ngrams(all_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_all_words_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used on all languages')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def all_words_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    all_D = pd.Series(str(all_text).split())
    top_20_all_words_trigrams = (pd.Series(nltk.ngrams(all_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_all_words_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used on all languages')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def python_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    python_D = pd.Series(str(python_text).split())
    top_20_python_bigrams = (pd.Series(nltk.ngrams(python_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_python_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used on Python')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def python_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    python_D = pd.Series(str(python_text).split())
    top_20_python_bigrams = (pd.Series(nltk.ngrams(python_D, 2))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_python_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used on Python')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def python_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    python_D = pd.Series(str(python_text).split())
    top_20_python_trigrams = (pd.Series(nltk.ngrams(python_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_python_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used on Python')
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def python_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    python_D = pd.Series(str(python_text).split())
    top_20_python_trigrams = (pd.Series(nltk.ngrams(python_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_python_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used on Python')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()


def javascript_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_bigrams = (pd.Series(nltk.ngrams(javascript_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_javascript_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used JavaScript')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def javascript_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_bigrams = (pd.Series(nltk.ngrams(javascript_D, 2))
                          .value_counts()
                          .head(20))
    # Top 2 Words Used in README(s)
    top_20_javascript_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used on Javascript')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def javascript_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_trigrams = (pd.Series(nltk.ngrams(javascript_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_javascript_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used on Javascript')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def javascript_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_trigrams = (pd.Series(nltk.ngrams(javascript_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_javascript_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used JavaScript')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def typescript_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_bigrams = (pd.Series(nltk.ngrams(typescript_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_typescript_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used on TypeScrypt')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def typescript_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_bigrams = (pd.Series(nltk.ngrams(typescript_D, 2))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_typescript_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used on TypeScript')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def typescript_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_trigrams = (pd.Series(nltk.ngrams(typescript_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_typescript_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used on TypeScript')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def typescript_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_trigrams = (pd.Series(nltk.ngrams(typescript_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_typescript_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used on Typescript')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def java_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    java_D = pd.Series(str(java_text).split())
    top_20_java_bigrams = (pd.Series(nltk.ngrams(java_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_java_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used on Java')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def java_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    java_D = pd.Series(str(java_text).split())
    top_20_java_bigrams = (pd.Series(nltk.ngrams(java_D, 2))
                          .value_counts()
                          .head(20))
    # Top 2 Words Used in README(s)
    top_20_java_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used on Java')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def java_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    java_D = pd.Series(str(java_text).split())
    top_20_java_trigrams = (pd.Series(nltk.ngrams(java_D, 3))
                                  .value_counts()
                                  .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_java_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used on Java')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def java_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    java_D = pd.Series(str(java_text).split())
    top_20_java_trigrams = (pd.Series(nltk.ngrams(java_D, 3))
                                  .value_counts()
                                  .head(20))
    # Top 3 Words Used in README(s)
    top_20_java_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used on Java')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def go_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    go_D = pd.Series(str(go_text).split())
    top_20_go_bigrams = (pd.Series(nltk.ngrams(go_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_go_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used on Go')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def go_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    go_D = pd.Series(str(go_text).split())
    top_20_go_bigrams = (pd.Series(nltk.ngrams(go_D, 2))
                          .value_counts()
                          .head(20))
    # Top 2 Words Used in README(s)
    top_20_go_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used on Go')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def go_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    go_D = pd.Series(str(go_text).split())
    top_20_go_trigrams = (pd.Series(nltk.ngrams(go_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_go_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used on Go')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def go_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    go_D = pd.Series(str(go_text).split())
    top_20_go_trigrams = (pd.Series(nltk.ngrams(go_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_go_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used on Go')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def unique_word_count():
    """ This function counts all the unique words that are in each read me and shows which coding language has the most in a barplot"""
    # Visual of the unique word counts
    train['unique_word_counts'] = train.lemmatized.apply(lambda x : len(set(x.split())))

    unique_train = pd.DataFrame(train.groupby('language').unique_word_counts.mean().sort_values(ascending=False)).reset_index()
    unique_train.head()

    ax = sns.barplot(x='language',y='unique_word_counts',data=unique_train)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()

def min_words_in_read_mes():
    """ This function does a value count for the amount of words found in each read me and returns the values of the lowest ones """
    # min README(s)
    train.lemmatized.apply(len).sort_values().head(20).plot.bar(x=train.repo)
    plt.title('Bottom 20 Shortest README Files')
    plt.xlabel('Repository (Index Number)')
    plt.ylabel('')
    plt.show()

def read_me_lengths():
    """ This fucntion allows us to vizulize the average read me lengths by word counts from largest to smallest based on coding languages """
    # Get the length of text in each README
    df['text_length'] = df.lemmatized.apply(len)
    #plots the mean of Readme lengths by programming language

    df.groupby(['language']).text_length.mean().sort_values(ascending = False).plot.bar()

    #inputs the chart title
    plt.title("Mean of Readme Lengths by Programming Language")
    #sets the size of the chart
    plt.rcParams["figure.figsize"] = (10, 5)