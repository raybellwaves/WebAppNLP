from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# import HashingVectorizer from local dir
from vectorizer import vect

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
# Unpickle classifier
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
    # Return the predicted label and probability
    label = {0: 'negative', 1: 'positive'}
    # Create text color to display positive in green and negative in red
    text_col = {0: 'red', 1: 'green'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], text_col[y], proba

def train(document, y):
    # Update classifier
    # Note resets when restarts the app    
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    # Store data for movie review
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

######## Flask
class ReviewForm(Form):
    # Render TextAreaField in reviewform.html (landing page)    
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])
    # Review must contain 15 characters

@app.route('/')
def index():
    # Render ReviewForm in index.html
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    # Fetch contents of submitted web form and pass to clf
    # Display the results in the renderd results.html
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, color, proba = classify(review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                font_color=color,
                                probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    # Fetch the predicted class label from results.html and update
    # based on user feedback
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    # Update the label in the train function
    train(review, y)
    # Create a new entry in the SQL lite database
    sqlite_entry(db, review, y)
    # Render thanks.html to thank the user for feedback
    return render_template('thanks.html',
                           content=review)

if __name__ == '__main__':
    app.run(debug=True)