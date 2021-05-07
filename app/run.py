import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
import joblib
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
app = Flask(__name__)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/disaster_response_db.db')
df = pd.read_sql_table('disaster_response_db_table', engine)

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

    category_names = df.iloc[:, 4:].columns
    category_boolean = (df.iloc[:, 4:] != 0).sum().values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    customscale = [[0, "rgb(160, 50, 30)"],
                   [0.1, "rgb(255, 40, 0)"],
                   [0.9, "rgb(200, 50, 0)"],
                   [1.0, "rgb(100, 0, 255)"]]
    # top words
    # category top 10 prep
    category_counts = df.iloc[:, 4:].sum(axis=0).sort_values(ascending=False)
    category_top = category_counts.head(10)
    category_names = list(category_top.index)

    # category frequencies prep
    labels = df.iloc[:, 4:].sum().sort_values(ascending=False).reset_index()
    labels.columns = ['category', 'count']
    label_values = labels['count'].values.tolist()
    label_names = labels['category'].values.tolist()

    graphs = [
        # GRAPH 1 - genre graph
        {

            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    #marker = dict(color=genre_counts,colorscale='viridis')
                    marker = dict(color=genre_counts,colorscale=customscale)
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

        # GRAPH 2 - category graph
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean,
                    marker = dict(color=category_boolean,colorscale='viridis')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Cout"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }

            }
        },
        # GRAPH 3 - Most Frequent Words
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_top
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_values,
                    marker=dict(color=category_boolean, colorscale='viridis')
                )
            ],

            'layout': {
                'title': "Messages categories frequency",
                'yaxis': {
                    'title': "Message Category Frequency"
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
    # app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
