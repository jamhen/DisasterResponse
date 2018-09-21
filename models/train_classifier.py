# import libraries
# download necessary NLTK data
import nltk
nltk.download(['punkt','stopwords'])

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
# import statements
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

# Load data files
def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterTable', engine)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = (df.columns[4:]).tolist()
    return X, Y, category_names


def tokenize(text):
    # tokenize messages
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmatized = [WordNetLemmatizer().lemmatize(w) for w in words]

    clean_words = []
    for tokens in lemmatized:
        clean_words.append(tokens.lower().strip()) 
    return clean_words


def build_model():
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42), n_jobs=-1)),
    ])
    # set tuning parameters
    parameters = {
            'tfidf__norm': ['l1', 'l2'],
            'clf__estimator__criterion':['gini', 'entropy']
    }
    # get uptimised model with grid search
    model = GridSearchCV(pipeline, param_grid=parameters,
                    cv=2, n_jobs=-1, verbose=1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    # predict model performance  over test set
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, 
                                target_names=category_names))
    # put test labels and predictions in pandas DataFrames
    test_df = pd.DataFrame(Y_test, columns=category_names)
    preds_df = pd.DataFrame(y_pred, columns=category_names)
    
    # print accuracy score for each category
    print("Accuracy Score for each Category")
    print("--------------------------------")    
    for category in category_names:
        accuracy = accuracy_score(test_df[category], preds_df[category])
        print("Accuracy score for {}: {:.3f} ".format(category, accuracy))
    return


def save_model(model, model_filepath):
    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
