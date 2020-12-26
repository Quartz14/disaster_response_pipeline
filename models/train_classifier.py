# import libraries
from functools import partial
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import nltk
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skm

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

lemmatizer = WordNetLemmatizer() #Initializes the lemmatizer
stop_words = set(stopwords.words('english')) #Gets the list of stop words from library
def tokenize(text):
    """Cleans and splits the given text into tokens

    Parameters
    ----------
    text : str
        The text/sentence to be tokenised

    Returns
    -------
    list
        a list of tokens
    """

    text_ = re.sub('\W',' ',text.lower())
    text_ = word_tokenize(text_)
    tokens = [lemmatizer.lemmatize(w).strip() for w in text_ if not w in stop_words]
    return(tokens)


def load_data(db_path):
    """Loads the datasets from the database

    Parameters
    ----------
    db_path : str
        The path of the database

    Returns
    -------
    X,Y
        The text messages(X) and classified categories(Y) are returned
    """

    engine = create_engine('sqlite:///'+db_path)
    connection = engine.connect()
    df = pd.read_sql_table('messages',connection)
    X = df['message']
    category_cols = []
    for col in df.columns:
        if(df[col].dtype=='int64') and (col!='id'):
            category_cols.append(col)
    Y =df[category_cols]
    Y = Y.replace(2, 0)
    # The child_alone category has no positive examples, hence that category is also ignored
    Y = Y.drop(['child_alone'],axis=1)
    return X,Y

def build_model():
    """Builds the model for classification of text

    Returns
    -------
    cv
        The model that is built, is returned
    """

    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=partial(tokenize))),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])

    parameters = {
    'vect__ngram_range':((3, 3), (1, 2)),
    'clf__estimator__n_estimators': [150,200],
    'clf__estimator__warm_start':[True,False]
    }
    pipeline_cv = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline_cv

def split_data(X,Y,test_size=0.3):
    """Splits the data into train and test sets

    Parameters
    ----------
    X,Y : dataframes
        The dependent and independent datasets, to be used for classification
    test_size : Float (optional)
        The ratio of test size to train size. Default is 0.3 -> 30% test data

    Returns
    -------
    X_train, X_test, y_train,y_test
        The split train and test datasets
    """

    X_train, X_test, y_train,y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train,y_test

def train_model(pipeline,xtrain, ytrain):
    """Trains the model with the train datasets

    Parameters
    ----------
    pipeline : scikitlearn pipeline
        The model returned upon calling the build_model function
    xtrain, ytrain : dataframes
        The train datasets returned from the split_data function

    Returns
    -------
    pipeline
        The model after training on the datasets is returned
    """
    stop_words = set(stopwords.words('english'))
    pipeline.fit(xtrain, ytrain)
    return pipeline

def test_model(pipeline, xtest,ytest):
    """Tests the trained model on the test data and prints accuracy,
    classification report, F1 score

    Parameters
    ----------
    pipeline : scikitlearn pipeline
        The model returned upon calling the train_model function
    xtest, ytest : dataframes
        The test datasets returned from the split_data function
    """

    predicted = pipeline.predict(xtest)
    acc = np.mean(predicted == ytest)
    print('Accuracy: ',acc)
    f1 = skm.f1_score(ytest, predicted,  average='micro')
    print('F1: ',f1)
    cm = skm.multilabel_confusion_matrix(ytest, predicted)
    print("_________________Classification matrix:_________________________")
    print(cm)
    print("__________________end________________________")
    print("Classification report")
    print( skm.classification_report(ytest,predicted))


def predict(pipeline, xtest):
    """Gets the prediction for a given text, using the trained model

    Parameters
    ----------
    pipeline : scikitlearn pipeline
        The model returned upon calling the train_model function
    xtest : dataframes
        The test dataset (messages) returned from the split_data function

    Returns
    -------
    prediction
        The prediction for the given test inputs using the given trained model
    """
    prediction = pipeline.predict(xtest)
    return prediction

def save_model(model, model_path):
    """Saves the trained model

    Parameters
    ----------
    model : scikitlearn pipeline
        The model returned upon calling the train_model function
    model_path: str
        The path where the model is to be stored

    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    dump(model, model_path) #F1:  0.509
    #clf = load('delete.pkl')


def main():
    """
    Main function that sequentiall call the above functions to build
    a disaster message classifying model

    """

    if len(sys.argv) == 3:
        #Obtaining the database and model file paths
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y= load_data(database_filepath)

        print("Spliting data into train,test")
        X_train, X_test, Y_train, Y_test = split_data(X,Y,test_size=0.3)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model = train_model(model,X_train, Y_train)

        print('Evaluating model...')
        test_model(model, X_test, Y_test)


        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
