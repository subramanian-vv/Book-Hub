# BookHub 

A user-based and product-based recommendation system that recommends books according to the users history or based on similarity in ratings.  

## How to run the system

1. Download the dataset from the link : http://jmcauley.ucsd.edu/data/amazon/
2. Run the following command to install the requirements
    ``` pip install requirements.txt ```
3. Run the `Recommendation.py` file to train the declared models and generate the required pickle file as follows 
    ```python3 Recommendation.py [model_name]```
    where, model_name is either SVD, KNN, or SVD2 where SVD2 uses Grid Search SV for parameter tuning.
4. Run the Flask application using the following command
    ``` python3 app.py ```
5. Open the following local-host link on your browser. 
    ``` localhost:5001 ```