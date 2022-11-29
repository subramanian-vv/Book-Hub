from flask import Flask, redirect, url_for, render_template, request
import numpy as np
import pandas as pd
import pickle

books = pickle.load(open('books.pkl','rb'))
popular_books = pickle.load(open('popular_books.pkl','rb'))
pivot = pickle.load(open('pivot.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))
user_ratings = pickle.load(open('user_recommendation.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    print(user_ratings.keys())
    return redirect(url_for('book_recommend'))

@app.route('/book_recommend')
def book_recommend():
    return render_template('book_recommend.html', books = list(popular_books['Book-Title'].values))

@app.route('/recommend_books')
def recommend_books():
    return render_template('book_recommend.html', books = list(popular_books['Book-Title'].values))

@app.route('/user_recommend')
def user_recommend():
    return render_template('user_recommend.html', users = list(user_ratings.keys()))

@app.route('/recommend_user_books')
def recommend_user_books():
    return render_template('user_recommend.html', users = list(user_ratings.keys()))

@app.route('/recommend_books',methods=['post'])
def book_based_recommendation():
    user_input = request.form.get('user_input')
    try:
        index = np.where(pivot.index == user_input)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse = True)[1:5]

        data = []
        for i in similar_items:
            item = []
            book = books[books['Book-Title'] == pivot.index[i[0]]]
            item.extend(list(book.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(book.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(book.drop_duplicates('Book-Title')['Image-URL-M'].values))
            data.append(item)
        print(data)

        return render_template('book_recommend.html', data = data, user_input = user_input, books = list(popular_books['Book-Title'].values))

    except:
        print("Error!")
        return render_template('book_recommend.html', user_input = user_input, flag = 1)

@app.route('/recommend_user_books', methods=['POST'])
def user_based_recommendation():
    user_input = request.form.get('user_input')
    try:
        data = []
        count = 0
        for i in user_ratings[user_input]:
            item = []
            item.extend(books[books["ISBN"] == i[0]]["Book-Title"].values)
            item.extend(books[books["ISBN"] == i[0]]["Book-Author"].values)
            item.extend(books[books["ISBN"] == i[0]]["Image-URL-M"].values)
            if count < 4:
                data.append(item)
            count += 1
        
        return render_template('user_recommend.html', data = data, user_input = user_input, users = list(user_ratings.keys()))

    except:
        print("Error!")
        return render_template('user_recommend.html', user_input = user_input, flag = 1)

if __name__ == '__main__':
    app.run(debug = True, port = 5001)