
#  Problem Statement: Recommendation system for products on an e-commerce website like Amazon.com. 

#  Load necessary libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import simplejson as json
import matplotlib.pyplot as plt
import warnings
import sklearn.metrics as metric
from math import sqrt
from collections import defaultdict
import pickle
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
from surprise import KNNBasic, SVD, NormalPredictor, KNNBaseline,KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering, Reader, dataset, accuracy
import surprise
from surprise import KNNWithMeans
from surprise.model_selection import GridSearchCV
from surprise import Dataset
from surprise import accuracy
from surprise import Reader

from sklearn.model_selection import train_test_split as normal_split
from surprise.model_selection import train_test_split 
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split

#*####################################################################################################################################################*#

# Book based recommendation

#*####################################################################################################################################################*#

def recommend_book_based(book_name, pt, similarity_scores, books):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data

def book_based_reco():

    books = pd.read_csv('books.csv')
    users = pd.read_csv('users.csv')
    ratings = pd.read_csv('ratings.csv')
    print(books.shape)
    print(ratings.shape)
    print(users.shape)

    # ## Popularity Based Recommender System


    ratings_with_name = ratings.merge(books,on='ISBN')

    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
    avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
    avg_rating_df.rename(columns={'Book-Rating':'avg_rating'},inplace=True)


    popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
    popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)
    popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]

    # Collaborative Filtering Based Recommender System

    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 50
    valid_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(valid_users)]
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
    pt.fillna(0,inplace=True)


    similarity_scores = cosine_similarity(pt)


    recommend_book_based('1984', pt, similarity_scores, books)
    books.drop_duplicates('Book-Title')

    pickle.dump(popular_df,open('popular_books.pkl','wb'))
    pickle.dump(pt,open('pivot.pkl','wb'))
    pickle.dump(books,open('books.pkl','wb'))
    pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))


#*####################################################################################################################################################*#

# User based recommendation

#*####################################################################################################################################################*#

# # Read and explore the dataset  ( Rename column/add headers, plot histograms, find data characteristics) 
def inputData(path):
    columns = ['userID', 'productID', 'ratings','timestamp']
    recomm_df = pd.read_csv(path)
    recomm_df.columns = columns
    recomm_df['ratings']  = recomm_df['ratings'].astype(int)
    #  Dropping the "timestamp" as it is not a needed field
    # Missing Value
    print(recomm_df.isna().sum())
    print(recomm_df.shape)
    return recomm_df
    
def printDetails(file):
    print(file.info())
    print(file.head())
    print(file.shape)
    print(file.describe().T)

# # plot histograms
def plotHistograms(recomm_df):
    recomm_df.hist('ratings',bins = 10)
    popular = recomm_df[['userID','ratings']].groupby('userID').sum().reset_index()
    popular_20 = popular.sort_values('ratings', ascending=False).head(n=20)

    objects = (list(popular_20['userID']))
    y_pos = np.arange(len(objects))
    performance = list(popular_20['ratings'])
    
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('userID')
    plt.title('Most popular')
    
    plt.show()

def printStats(recomm_df):
    recomm_df.userID.value_counts()
    print('Number of unique users', len(recomm_df['userID'].unique()))
    print('Number of unique products', len(recomm_df['productID'].unique()))
    print('Unique Ratings', recomm_df['ratings'].unique())
    min_ratings1 = recomm_df[(recomm_df['ratings'] < 2.0)]
    print('Number of unique products rated low',len(min_ratings1['productID'].unique()))
    med_ratings1 = recomm_df[(recomm_df['ratings'] > 2.0) & (recomm_df['ratings'] < 4.0)]
    print('Number of unique products rated medium',len(med_ratings1['productID'].unique()))
    max_ratings1 = recomm_df[recomm_df['ratings'] >= 4.0]
    print('Number of unique products rated high',len(max_ratings1['productID'].unique()))
    avg_rating_prod = recomm_df.groupby('productID').sum() / recomm_df.groupby('productID').count()
    avg_rating_prod.drop('userID', axis=1,inplace =True)
    print ('Top 10 highly rated products \n',avg_rating_prod.nlargest(10,'ratings'))

def preprocessData(recomm_df):
    # Take a subset of the dataset to make it less sparse/ denser. 
    # ( For example, keep the users only who has given 50 or more number of ratings )
    userID = recomm_df.groupby('userID').count()
    top_user = userID[userID['ratings'] >= 50].index
    topuser_ratings_df = recomm_df[recomm_df['userID'].isin(top_user)]
    print(topuser_ratings_df.shape)
    topuser_ratings_df.sort_values(by='ratings', ascending=False).head()

    # Keep data only for products that have 50 or more ratings
    prodID = recomm_df.groupby('productID').count()
    top_prod = prodID[prodID['ratings'] >= 50].index
    top_ratings_df = topuser_ratings_df[topuser_ratings_df['productID'].isin(top_prod)]
    top_ratings_df.sort_values(by='ratings', ascending=False).head()
    top_ratings_df.shape
    return top_user, top_ratings_df

# Build Popularity Recommender model.
def mean_based_reco(top_ratings_df):
    # Split the data randomly into train and test dataset. ( For example, split it in 70/30 ratio)
    train_data, test_data = normal_split(top_ratings_df, test_size = 0.30, random_state=0)
    print('Train data shape : ',train_data.shape)
    print('Test data shape : ',test_data.shape)
    #Building the recommendations based on the average of all user ratings for each product.
    train_data_grouped = train_data.groupby('productID').mean().reset_index()
    train_data_grouped.head()
    train_data_sort = train_data_grouped.sort_values(['ratings', 'productID'], ascending=False)
    train_data_sort.head()
    train_data.groupby('productID')['ratings'].count().sort_values(ascending=False).head(10) 
    ratings_mean_count = pd.DataFrame(train_data.groupby('productID')['ratings'].mean()) 
    ratings_mean_count['rating_counts'] = pd.DataFrame(train_data.groupby('productID')['ratings'].count())  
    ratings_mean_count.head()  
    pred_df = test_data[['userID', 'productID', 'ratings']]
    pred_df.rename(columns = {'ratings' : 'true_ratings'}, inplace=True)
    pred_df = pred_df.merge(train_data_sort, left_on='productID', right_on = 'productID')
    pred_df.head(3)
    pred_df.rename(columns = {'ratings' : 'predicted_ratings'}, inplace = True)
    pred_df.head()

    MSE = metric.mean_squared_error(pred_df['true_ratings'], pred_df['predicted_ratings'])
    print('The RMSE value for Popularity Recommender model is', sqrt(MSE))
    
    return pred_df


# # Build Collaborative Filtering model
def collaborativeFiltering(option, top_ratings_df):
    # Converting Pandas Dataframe to Surpise format
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(top_ratings_df[['userID', 'productID', 'ratings']],reader)
    # Split data to train and test
    
    trainset, testset = train_test_split(data, test_size=.3,random_state=0)
    # Evaluate all the models. ( Once the model is trained on the training data, it can be used to compute 
    # the error (like RMSE) on 
    # predictions made on the test data.) You can also use a different method to evaluate the models.
    if option.lower() == 'knn':
        # KNNWithMeans
        algo_user = KNNWithMeans(k=10, min_k=6, sim_options={'name': 'pearson_baseline', 'user_based': True})
        algo_user.fit(trainset)
        # Evalute on test set
        test_pred = algo_user.test(testset)
        test_pred[0]
        # compute RMSE
        print(accuracy.rmse(test_pred)) #range of value of error
    elif option.lower() == 'svd':
        svd_model = SVD(n_factors=50,reg_all=0.02)
        svd_model.fit(trainset)
        test_pred = svd_model.test(testset)
        # svd_model.summary()
        # compute RMSE
        print(accuracy.rmse(test_pred))
    else:
        #  Parameter tuning of SVD Recommendation system
        param_grid = {'n_factors' : [5,10,15], "reg_all":[0.01,0.02]}
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3,refit = True)
        gs.fit(data)
        # get best parameters
        gs.best_params
        # Use the "best model" for prediction
        test_pred = gs.test(testset)
        print(accuracy.rmse(test_pred))
    return test_pred


# Get top - K ( K = 5) recommendations. 
# Since our goal is to recommend new products to each user based on his/her habits, we will recommend 5 new products.
def get_top_n(predictions, n=5):
  
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    json_data = json.dumps(top_n, indent=2)
    print(json_data)
    with open('user_recommendation.pkl', 'wb') as fp:
        pickle.dump(top_n, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return top_n

def getUserReco(user_id):
    with open('user_recommendation.pkl', 'rb') as fp:
        data = pickle.load(fp)
    print(data[user_id])
            



if __name__ == "__main__":
    path = './ratings_Books.csv'
    recomm_df = inputData(path)
    printDetails(recomm_df)
    plotHistograms(recomm_df)
    printStats(recomm_df)
    top_user, top_ratings_df = preprocessData(recomm_df)
    top_ratings_df.head()
    pred_df = mean_based_reco(top_ratings_df)
    test_pred = collaborativeFiltering('svd-2', top_ratings_df)
    top_n = get_top_n(test_pred, n=5)
    # Print the recommended items for each user
    getUserReco('ANIZRIJEUTRXN')

    # Now for book based recommendation 
    book_based_reco()




# # Summarise your insights

