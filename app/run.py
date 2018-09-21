import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    related_category1 =[ df.loc[df['weather_related']==1].loc[df['genre']=='direct'].count()['message'], 
                         df.loc[df['weather_related']==1].loc[df['genre']=='news'].count()['message'], 
                         df.loc[df['weather_related']==1].loc[df['genre']=='social'].count()['message']]
    related_category0 =[ df.loc[df['weather_related']==0].loc[df['genre']=='direct'].count()['message'], 
                         df.loc[df['weather_related']==0].loc[df['genre']=='news'].count()['message'], 
                         df.loc[df['weather_related']==0].loc[df['genre']=='social'].count()['message']]
    
    categories = df.columns[4:].tolist()
    received_msgs = df.iloc[:, 4:].sum().tolist()

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x = genre_names,
                    y = related_category1,
                    name = "Weather Related",
                    marker = dict(color='green')
                    ),
                Bar(
                    x = genre_names,
                    y = related_category0,
                    name = "Not Weather Related",
                    marker = dict(color='red')
                    )
            ],

            'layout': {
                'title': "Distribution of Weather Related Messages with Genre",
                'yaxis': {
                    'title':"Weather Messages Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
        {
            'data': [
                Histogram(
                    x = categories,
                    y = received_msgs,
                    histfunc = 'sum',
                    marker = dict(color='green')
                )                
            ],

            'layout': {
                'title': "Histogram Chart Frequency of Categories of Messages",
                'yaxis': {
                    'title':"Message Category Frequency"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
