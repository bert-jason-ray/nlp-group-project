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
from mergedeep import merge

# Starting our exploration we quickly noticed a trend of a handful of programing languages dominating our data.
# As a result, we made the decision to focus on these main languages in an effort build a model more accurately discern between them
def programing_language_distribution():
    sns.set_theme(style="white")
    ax = sns.countplot(x="language", data=train, palette="Set3",order = train['language'].value_counts().index)
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('Programming Language (Readme.md)', size = 16)
    ax.set_ylabel('Count / Frequency', size = 16)
    ax.set_title("Programming Language Distribution", size = 20)
    plt.show()
programing_language_distribution()

# Set up word counts dataframe
all_text = ' '.join(train.lemmatized)
javascript_text = ' '.join(train[train.language == 'JavaScript'].lemmatized)
python_text = ' '.join(train[train.language == 'Python'].lemmatized)
typescript_text = ' '.join(train[train.language == 'TypeScript'].lemmatized)
java_text = ' '.join(train[train.language == 'Java'].lemmatized)
go_text = ' '.join(train[train.language == 'Go'].lemmatized)
kotlin_text = ' '.join(train[train.language == 'Kotlin'].lemmatized)

def all_top_words():
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
    # Visualize word distribution
    word_counts.sort_values(by='all', ascending=False)[['all','JavaScript', 'python', 'typescript','java', 'go', 'kotlin']].head(9).plot.bar()
    plt.title('Words Used Most in README Files')
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('Nine Most Common Words Used')
    plt.ylabel('Word Quantity')
    plt.show()

def all_words_bigram_wordcloud():
    all_D = pd.Series(str(all_text).split())
    top_20_all_words_bigrams = (pd.Series(nltk.ngrams(all_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_all_words_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def all_words_bigram_barplot():
    all_D = pd.Series(str(all_text).split())
    top_20_all_words_bigrams = (pd.Series(nltk.ngrams(all_D, 2))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_all_words_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def all_words_trigram_wordcloud():
    all_D = pd.Series(str(all_text).split())
    top_20_all_words_trigrams = (pd.Series(nltk.ngrams(all_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_all_words_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def all_words_trigram_barplot():
    all_D = pd.Series(str(all_text).split())
    top_20_all_words_trigrams = (pd.Series(nltk.ngrams(all_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_all_words_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def python_bigram_wordcloud():
    python_D = pd.Series(str(python_text).split())
    top_20_python_bigrams = (pd.Series(nltk.ngrams(python_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_python_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def python_bigram_barplot():
    python_D = pd.Series(str(python_text).split())
    top_20_python_bigrams = (pd.Series(nltk.ngrams(python_D, 2))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_python_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def python_trigram_wordcloud():
    python_D = pd.Series(str(python_text).split())
    top_20_python_trigrams = (pd.Series(nltk.ngrams(python_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_python_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def python_trigram_barplot():
    python_D = pd.Series(str(python_text).split())
    top_20_python_trigrams = (pd.Series(nltk.ngrams(python_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_python_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()
python_trigram_wordcloud()
python_trigram_barplot()

def javascript_bigram_wordcloud():
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_bigrams = (pd.Series(nltk.ngrams(javascript_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_javascript_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def javascript_bigram_barplot():
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_bigrams = (pd.Series(nltk.ngrams(javascript_D, 2))
                          .value_counts()
                          .head(20))
    # Top 2 Words Used in README(s)
    top_20_javascript_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def javascript_trigram_wordcloud():
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_trigrams = (pd.Series(nltk.ngrams(javascript_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_javascript_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def javascript_trigram_barplot():
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_trigrams = (pd.Series(nltk.ngrams(javascript_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_javascript_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def typescript_bigram_wordcloud():
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_bigrams = (pd.Series(nltk.ngrams(typescript_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_typescript_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def typescript_bigram_barplot():
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_bigrams = (pd.Series(nltk.ngrams(typescript_D, 2))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_typescript_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def typescript_trigram_wordcloud():
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_trigrams = (pd.Series(nltk.ngrams(typescript_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_typescript_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def typewscript_trigram_barplot():
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_trigrams = (pd.Series(nltk.ngrams(typescript_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_typescript_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def java_bigram_wordcloud():
    java_D = pd.Series(str(java_text).split())
    top_20_java_bigrams = (pd.Series(nltk.ngrams(java_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_java_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
def java_bigram_barplot():
    java_D = pd.Series(str(java_text).split())
    top_20_java_bigrams = (pd.Series(nltk.ngrams(java_D, 2))
                          .value_counts()
                          .head(20))
    # Top 2 Words Used in README(s)
    top_20_java_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def java_trigram_wordcloud():
    java_D = pd.Series(str(java_text).split())
    top_20_java_trigrams = (pd.Series(nltk.ngrams(java_D, 3))
                                  .value_counts()
                                  .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_java_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
def java_trigram_barplot():
    java_D = pd.Series(str(java_text).split())
    top_20_java_trigrams = (pd.Series(nltk.ngrams(java_D, 3))
                                  .value_counts()
                                  .head(20))
    # Top 3 Words Used in README(s)
    top_20_java_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def go_bigram_wordcloud():
    go_D = pd.Series(str(go_text).split())
    top_20_go_bigrams = (pd.Series(nltk.ngrams(go_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_go_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 2 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
def go_bigram_barplot():
    go_D = pd.Series(str(go_text).split())
    top_20_go_bigrams = (pd.Series(nltk.ngrams(go_D, 2))
                          .value_counts()
                          .head(20))
    # Top 2 Words Used in README(s)
    top_20_go_bigrams.plot.bar()
    plt.title('Top 2 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def go_trigram_wordcloud():
    go_D = pd.Series(str(go_text).split())
    top_20_go_trigrams = (pd.Series(nltk.ngrams(go_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_go_trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(8, 4))
    plt.title('Top 3 Word clusters Used')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
def go_trigram_barplot():
    go_D = pd.Series(str(go_text).split())
    top_20_go_trigrams = (pd.Series(nltk.ngrams(go_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_go_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used')
    plt.xlabel('words')
    plt.ylabel('')
    plt.show()

def unique_word_count():
    # Visual of the unique word counts
    train['unique_word_counts'] = train.lemmatized.apply(lambda x : len(set(x.split())))

    unique_train = pd.DataFrame(train.groupby('language').unique_word_counts.mean().sort_values(ascending=False)).reset_index()
    unique_train.head()

    ax = sns.barplot(x='language',y='unique_word_counts',data=unique_train)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()

def min_words_in_read_mes():
    # min README(s)
    train.lemmatized.apply(len).sort_values().head(20).plot.bar(x=train.repo)
    plt.title('Bottom 20 Shortest README Files')
    plt.xlabel('Repository (Index Number)')
    plt.ylabel('')
    plt.show()