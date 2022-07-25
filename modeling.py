import pandas as pd
import numpy as np
import acquire
import prepare

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, recall_score, plot_confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

def nlp_X_split(X_data, y_data):
    # df = prepare.prep_github_data(df, column='readme_contents')
    # Create the tf-idf model
    #tfidf = TfidfVectorizer()
    #model = df.copy
    #top_languages = ['JavaScript', 'Python', 'TypeScript']
    #model[language] = model.language.apply(lambda lang : lang if lang in top_languages else 'not_top')
    # Fit the model and create the X, y variables for modeling
    #X = tfidf.fit_transform(model.lemmatized)
    #y = model.language 
    # Split the data into train (55%) validate (24%) test (20%) split
    X_train_validate, X_test, y_train_validate, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=.2, random_state = 123)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate, y_train_validate, stratify=y_train_validate, test_size=.3, random_state = 123)
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def test_model(X_train, y_train, X_validate, y_validate, model, model_name, score_df):
    '''
    Function takes in X and y train
    X and y validate (or test) 
    A model with it's hyper parameters
    And a df to store the scores 

    - Set up an empty dataframe with score_df first
    - score_df = pd.DataFrame(columns = ['model_name', 'train_score', 'validate_score', 'test_score'])
    '''
    this_model = model

    this_model.fit(X_train, y_train)

    # Check with Validate also added test

    train_score = this_model.score(X_train, y_train)
    
    validate_score = this_model.score(X_validate, y_validate)
    
    test_score = this_model.score(X_test, y_test)
    
    model_dict = {'model_name': model_name, 
                  'train_score':  round(train_score*100,2), 
                  'validate_score': round(validate_score*100,2),
                  'test_score': round(test_score*100,2)}
    score_df = score_df.append(model_dict, ignore_index = True)
    
    return score_df

def print_metrics(model, X, y, pred, language_names, set_name = 'This Set'):
    '''
    This function takes in a model, 
    X dataframe
    y dataframe 
    predictions 
    Class_names (aka ['Java', 'Javascript', 'Jupyter Notebook', 'PHP'])
    and a set name (aka train, validate or test)
    Prints out a classification report 
    and confusion matrix as a heatmap
    To customize colors change insdie the function

    - IMPORTANT change lables inside this function
    '''
    print(model)
    print(f"{set_name} Scores")
    print('Accuracy Score: {:.2%}'.format(accuracy_score(y, pred)))
    #print(round(accuracy_score*100,2)(y, pred))
    print(classification_report(y, pred))
    
    #purple_cmap = sns.cubehelix_palette(as_cmap=True)
    crest = sns.color_palette("crest", as_cmap=True)
    
    with sns.axes_style("white"):
        matrix = plot_confusion_matrix(model,X, y, display_labels=language_names, cmap = crest)
        plt.grid(False)
        plt.show()
        print()

def make_models_and_print_metrics(model, model_name, X_train, y_train, X_validate, y_validate, language_names):
    '''
    This function takes in a model object,
    Name for the model (for vis purposes)
    X_train, y_train
    X_validate and y_validate
    and the names of your classes (aka category names)
    Uses print metrics function 
    '''
    model.fit(X_train, y_train)

    #predict for train and validate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_validate)
    
    #see metrics for train
    model.print_metrics(model, X_train, y_train, train_pred, language_names, set_name='Train')
    #print metrics for validate
    model.print_metrics(model, X_validate, y_validate, val_pred, language_names, set_name='Validate')
    print('-------------------------------------------------------------------\n')