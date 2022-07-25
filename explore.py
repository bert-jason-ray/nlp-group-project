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
# Set up word counts from dataframe and naming them for the use in the follwing functions
# all combined languages
all_text = ' '.join(train.lemmatized)
#javascript
javascript_text = ' '.join(train[train.language == 'JavaScript'].lemmatized)
#python
python_text = ' '.join(train[train.language == 'Python'].lemmatized)
#typescript
typescript_text = ' '.join(train[train.language == 'TypeScript'].lemmatized)
#java
java_text = ' '.join(train[train.language == 'Java'].lemmatized)
#go
go_text = ' '.join(train[train.language == 'Go'].lemmatized)
#Kotlin
kotlin_text = ' '.join(train[train.language == 'Kotlin'].lemmatized)


 
def programing_language_distribution():
    """Starting our exploration we quickly noticed a trend of a handful of programing languages dominating our data.
    As a result, we made the decision to focus on these main languages in an effort build a model more accurately"""

    sns.set_theme(style="white")
    # Code that counts the number of times each vallue is used .
    ax = sns.countplot(x="language", data=train, palette="Set3",order = train['language'].value_counts().index)
    #axis of text on x axis
    ax.tick_params(axis='x', rotation=90)
    #label for x axis
    ax.set_xlabel('Programming Language (Readme.md)', size = 16)
    #y axis label
    ax.set_ylabel('Count / Frequency', size = 16)
    # title of graph
    ax.set_title("Programming Language Distribution", size = 20)
    # produce graph
    plt.show()

def all_top_words():
    """ This function allows us to vizualize the top 30 wordsfor all and top 30 words for each coding languages"""
    all_freq = pd.Series(str(all_text).split()).value_counts()
    # split words from language and count them seperately
    javascript_freq = pd.Series(str(javascript_text).split()).value_counts()
    # split words from language and count them seperately
    python_freq = pd.Series(str(python_text).split()).value_counts()
    # split words from language and count them seperately
    typeScript_freq = pd.Series(str(typescript_text).split()).value_counts()
    # split words from language and count them seperately
    java_freq = pd.Series(str(java_text).split()).value_counts()
    # split words from language and count them seperately
    go_freq = pd.Series(str(go_text).split()).value_counts()
    # split words from language and count them seperately
    kotlin_freq = pd.Series(str(kotlin_text).split()).value_counts()
    # combines all languages under one dataframe
    word_counts = pd.concat([all_freq, javascript_freq, python_freq, typeScript_freq,java_freq, go_freq, kotlin_freq], sort=True, axis=1)
    # renaming collumns 
    word_counts.columns = ['all', 'JavaScript', 'python', 'typescript','java', 'go', 'kotlin']
    # function to count words and add them as integers
    word_counts = word_counts.fillna(0).apply(lambda s: s.astype(int))
    #top 30 of all languages
    top_30 = word_counts.sort_values(by='all', ascending=False).head(30)
    return top_30

def word_distribution_vizual():
    """ This function allows us to see the word distribution rates from all, python, javascipt, typescript, java, go, and kotlin coding laguages in a bar chart across all read mes"""
    # Set up word counts from dataframe and naming them for the use in the follwing functions
    # all combined languages
    all_text = ' '.join(train.lemmatized)
    #javascript
    javascript_text = ' '.join(train[train.language == 'JavaScript'].lemmatized)
    #python
    python_text = ' '.join(train[train.language == 'Python'].lemmatized)
    #typescript
    typescript_text = ' '.join(train[train.language == 'TypeScript'].lemmatized)
    #java
    java_text = ' '.join(train[train.language == 'Java'].lemmatized)
    #go
    go_text = ' '.join(train[train.language == 'Go'].lemmatized)
    #Kotlin
    kotlin_text = ' '.join(train[train.language == 'Kotlin'].lemmatized)
    #frequency of counts for all and individual coding languages
    all_freq = pd.Series(str(all_text).split()).value_counts()
    javascript_freq = pd.Series(str(javascript_text).split()).value_counts()
    python_freq = pd.Series(str(python_text).split()).value_counts()
    typeScript_freq = pd.Series(str(typescript_text).split()).value_counts()
    java_freq = pd.Series(str(java_text).split()).value_counts()
    go_freq = pd.Series(str(go_text).split()).value_counts()
    kotlin_freq = pd.Series(str(kotlin_text).split()).value_counts()
    #combining all word counts together into one dataframe.
    word_counts = pd.concat([all_freq, javascript_freq, python_freq, typeScript_freq,java_freq, go_freq, kotlin_freq], sort=True, axis=1)
    #naming collumns 
    word_counts.columns = ['all', 'JavaScript', 'python', 'typescript','java', 'go', 'kotlin']
    #fill na values with 0
    word_counts = word_counts.fillna(0).apply(lambda s: s.astype(int))
    # Visualize word distribution
    word_counts.sort_values(by='all', ascending=False)[['all','JavaScript', 'python', 'typescript','java', 'go', 'kotlin']].head(9).plot.bar()
    #title
    plt.title('Words Used Most in README Files')
    #style of graph
    plt.style.use('seaborn-whitegrid')
    #label for x axis
    plt.xlabel('Nine Most Common Words Used')
    #label for y axis
    plt.ylabel('Word Quantity')
    #produce graph
    plt.show()

def all_words_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    #using All Data words
    all_D = pd.Series(str(all_text).split())
    # top 20 bigrams 
    top_20_all_words_bigrams = (pd.Series(nltk.ngrams(all_D, 2))
                          .value_counts()
                          .head(20))
    #data fuction that summons top 20 words
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_all_words_bigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # size of graph
    plt.figure(figsize=(8, 4))
    # title of graph
    plt.title('Top 2 Word clusters Used on all languages')
    # summons words as images
    plt.imshow(img)
    # removes x and y axis
    plt.axis('off')
    #produce graph
    plt.show()

def all_words_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    #using all data
    all_D = pd.Series(str(all_text).split())
    #top 20 bigrams
    top_20_all_words_bigrams = (pd.Series(nltk.ngrams(all_D, 2))
                          .value_counts()
                          .head(20))
    # Top 20 Words Used in README(s)
    top_20_all_words_bigrams.plot.bar()
    #title for graph
    plt.title('Top 2 Word clusters Used on all languages')
    # label for x axis
    plt.xlabel('words')
    #label for y axis
    plt.ylabel('')
    # produce graph
    plt.show()

def all_words_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    #using all data
    all_D = pd.Series(str(all_text).split())
    #top 20 bigrams
    top_20_all_words_trigrams = (pd.Series(nltk.ngrams(all_D, 3))
                          .value_counts()
                          .head(20))
    # Top 20 Words Used in README(s)
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_all_words_trigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # size of graph
    plt.figure(figsize=(8, 4))
    # title of graph
    plt.title('Top 3 Word clusters Used on all languages')
    # summons words as images
    plt.imshow(img)
    # removes x and y axis
    plt.axis('off')
    #produce graph
    plt.show()

def all_words_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    #using all data
    all_D = pd.Series(str(all_text).split())
    #top 20 bigram
    top_20_all_words_trigrams = (pd.Series(nltk.ngrams(all_D, 3))
                          .value_counts()
                          .head(20))
    # Top 20 Words Used in README(s)
    top_20_all_words_trigrams.plot.bar()
    #title for graph
    plt.title('Top 3 Word clusters Used on all languages')
    # label for x axis
    plt.xlabel('words')
    #label for y axis
    plt.ylabel('')
    # produce graph
    plt.show()

def python_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    #using All Data words
    python_D = pd.Series(str(python_text).split())
    # top 20 bigrams 
    top_20_python_bigrams = (pd.Series(nltk.ngrams(python_D, 2))
                          .value_counts()
                          .head(20))
    #data function that summons top 20 words
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_python_bigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 2 Word clusters Used on Python')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()

def python_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    python_D = pd.Series(str(python_text).split())
    top_20_python_bigrams = (pd.Series(nltk.ngrams(python_D, 2))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_python_bigrams.plot.bar()
    #title
    plt.title('Top 2 Word clusters Used on Python')
    # x label
    plt.xlabel('words')
    # y label 
    plt.ylabel('')
    # produce graph
    plt.show()

def python_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    # count of words used seperately
    python_D = pd.Series(str(python_text).split())
    # top 20 word clusters
    top_20_python_trigrams = (pd.Series(nltk.ngrams(python_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_python_trigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 3 Word clusters Used on Python')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()


def python_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    python_D = pd.Series(str(python_text).split())
    top_20_python_trigrams = (pd.Series(nltk.ngrams(python_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_python_trigrams.plot.bar()
    #graph title
    plt.title('Top 3 Word clusters Used on Python')
    # x label
    plt.xlabel('words')
    # y label 
    plt.ylabel('')
    # produce graph
    plt.show()

def javascript_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_bigrams = (pd.Series(nltk.ngrams(javascript_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_javascript_bigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 2 Word clusters Used on JavaScript')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()

def javascript_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_bigrams = (pd.Series(nltk.ngrams(javascript_D, 2))
                          .value_counts()
                          .head(20))
    # Top 2 Words Used in README(s)
    top_20_javascript_bigrams.plot.bar()
    #title for graph
    plt.title('Top 2 Word clusters Used on JavaScript')
    # x label
    plt.xlabel('words')
    # y label 
    plt.ylabel('')
    # produce graph
    plt.show()
def javascript_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_trigrams = (pd.Series(nltk.ngrams(javascript_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_javascript_trigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 3 Word clusters Used on JavaScript')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()

def javascript_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    javascript_D = pd.Series(str(javascript_text).split())
    top_20_javascript_trigrams = (pd.Series(nltk.ngrams(javascript_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_javascript_trigrams.plot.bar()
    #title for graph
    plt.title('Top 3 Word clusters Used JavaScript')
    # x label
    plt.xlabel('words')
    # y label 
    plt.ylabel('')
    # produce graph
    plt.show()
def typescript_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_bigrams = (pd.Series(nltk.ngrams(typescript_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_typescript_bigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 2 Word clusters Used on TypeScript')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()

def typescript_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_bigrams = (pd.Series(nltk.ngrams(typescript_D, 2))
                          .value_counts()
                          .head(20))
    # Top 20 Words Used in README(s)
    top_20_typescript_bigrams.plot.bar()
    # graph title
    plt.title('Top 2 Word clusters Used on TypeScript')
    # x label
    plt.xlabel('words')
    # y label 
    plt.ylabel('')
    # produce graph
    plt.show()

def typescript_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_trigrams = (pd.Series(nltk.ngrams(typescript_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_typescript_trigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 3 Word clusters Used on TypeScript')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()

def typescript_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    typescript_D = pd.Series(str(typescript_text).split())
    top_20_typescript_trigrams = (pd.Series(nltk.ngrams(typescript_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_typescript_trigrams.plot.bar()
    plt.title('Top 3 Word clusters Used on TypeScript')
    # x label
    plt.xlabel('words')
    # y label 
    plt.ylabel('')
    # produce graph
    plt.show()
def java_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    java_D = pd.Series(str(java_text).split())
    top_20_java_bigrams = (pd.Series(nltk.ngrams(java_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_java_bigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 2 Word clusters Used on Java')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()

def java_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    java_D = pd.Series(str(java_text).split())
    top_20_java_bigrams = (pd.Series(nltk.ngrams(java_D, 2))
                          .value_counts()
                          .head(20))
    # Top 20 Words Used in README(s)
    top_20_java_bigrams.plot.bar()
    #graph title
    plt.title('Top 2 Word clusters Used on Java')
    # x label
    plt.xlabel('words')
    # y label 
    plt.ylabel('')
    # produce graph
    plt.show()

def java_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    java_D = pd.Series(str(java_text).split())
    top_20_java_trigrams = (pd.Series(nltk.ngrams(java_D, 3))
                                  .value_counts()
                                  .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_java_trigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 3 Word clusters Used on Java')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()

def java_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    java_D = pd.Series(str(java_text).split())
    top_20_java_trigrams = (pd.Series(nltk.ngrams(java_D, 3))
                                  .value_counts()
                                  .head(20))
    # Top 20 (3 Word cluster Used in README(s))
    top_20_java_trigrams.plot.bar()
    #graph title
    plt.title('Top 3 Word clusters Used on Java')
    # x label
    plt.xlabel('words')
    # y label 
    plt.ylabel('')
    # produce graph
    plt.show()

def go_bigram_wordcloud():
    """ allows us to see bigram(top two words used) clouds, shows us the more common used words as larger images"""
    go_D = pd.Series(str(go_text).split())
    top_20_go_bigrams = (pd.Series(nltk.ngrams(go_D, 2))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_go_bigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 2 Word clusters Used on Go')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()

def go_bigram_barplot():
    """ allows us to see bigram(top two words used) barplots, shows us the more common used words as larger bars"""
    go_D = pd.Series(str(go_text).split())
    top_20_go_bigrams = (pd.Series(nltk.ngrams(go_D, 2))
                          .value_counts()
                          .head(20))
    # Top 2 Words Used in README(s)
    top_20_go_bigrams.plot.bar()
    #graph title 
    plt.title('Top 2 Word clusters Used on Go')
    # x axis label
    plt.xlabel('words')
    # y axis label
    plt.ylabel('')
    # produce graph
    plt.show()

def go_trigram_wordcloud():
    """ allows us to see trigram(top 3 words used) clouds, shows us the more common used word groupings as larger images"""
    go_D = pd.Series(str(go_text).split())
    top_20_go_trigrams = (pd.Series(nltk.ngrams(go_D, 3))
                          .value_counts()
                          .head(20))
    data = {k[0] + ' ' + k[1]: v for k, v in top_20_go_trigrams.to_dict().items()}
    #style of imaging code
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    # graph size
    plt.figure(figsize=(8, 4))
    #title for graph
    plt.title('Top 3 Word clusters Used on Go')
    # label for x axis
    plt.imshow(img)
    #label for y axis
    plt.axis('off')
    # produce graph
    plt.show()

def go_trigram_barplot():
    """ allows us to see trigram(top 3 words used) barplors, shows us the more common used word groupings as larger bars"""
    go_D = pd.Series(str(go_text).split())
    top_20_go_trigrams = (pd.Series(nltk.ngrams(go_D, 3))
                          .value_counts()
                          .head(20))
    # Top 3 Words Used in README(s)
    top_20_go_trigrams.plot.bar()
    # graph title
    plt.title('Top 3 Word clusters Used on Go')
    # x axis lable
    plt.xlabel('words')
    # y axis label
    plt.ylabel('')
    # produce graph
    plt.show()

def unique_word_count():
    """ This function counts all the unique words that are in each read me and shows which coding language has the most in a barplot"""
    # Visual of the unique word counts
    train['unique_word_counts'] = train.lemmatized.apply(lambda x : len(set(x.split())))
    #add to dataframe 
    unique_train = pd.DataFrame(train.groupby('language').unique_word_counts.mean().sort_values(ascending=False)).reset_index()
    unique_train.head()
    # assignince y and x axis on graph
    ax = sns.barplot(x='language',y='unique_word_counts',data=unique_train)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    # produce graph
    plt.show()

def min_words_in_read_mes():
    """ This function does a value count for the amount of words found in each read me and returns the values of the lowest ones """
    # Min README(s)
    train.lemmatized.apply(len).sort_values().head(20).plot.bar(x=train.repo)
    # title for graph 
    plt.title('Bottom 20 Shortest README Files')
    # x axis label
    plt.xlabel('Repository (Index Number)')
    #y axis label
    plt.ylabel('')
    # produce graph
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