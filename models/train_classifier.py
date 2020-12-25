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

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def tokenize(text):
    text_ = re.sub('\W',' ',text.lower())
    text_ = word_tokenize(text_)
    tokens = [lemmatizer.lemmatize(w).strip() for w in text_ if not w in stop_words]
    return(tokens)


def load_data(db_path):
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
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=partial(tokenize))),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    return pipeline

def split_data(X,Y,test_size=0.3):
    X_train, X_test, y_train,y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train,y_test

def train_model(pipeline,xtrain, ytrain):
    stop_words = set(stopwords.words('english'))
    pipeline.fit(xtrain, ytrain)
    return pipeline

def test_model(pipeline, xtest,ytest):
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
    prediction = pipeline.predict(xtest)
    return prediction

def save_model(model, model_path):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    dump(model, model_path) #F1:  0.509
    #clf = load('delete.pkl')


def main():

    if len(sys.argv) == 3:
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
