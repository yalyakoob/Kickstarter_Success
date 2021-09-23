"""for loading pre-trained XGBClassifier, then generating predictions"""

import joblib
import numpy as np
import pandas as pd


xgb_model = joblib.load('xgb_model')


def init_model_input():
    """
    For a prediction of funding success v. failure, app user may input feature values manually, via the app's HTML routes; but most features' values will remain as defined by this reset-type function.

    This init_model_input() will not be called directly -- it is a helper function called within generate_prediction().
    """

    goal_usd = 5000.                     # median = 5000
    
    campaign_duration_in_days = int(30)  # median = 30
    
    category_art, category_music, category_crafts, category_comics, category_food, category_publishing, category_technology, category_journalism, category_design, category_film_and_video, category_theater, category_games, category_photography, category_fashion, country_GB, country_US, country_CA, country_other = np.zeros(18, dtype=np.int64) # all one-hot encoded feature levels are preset to 0

    feature_vector = pd.Series([goal_usd, campaign_duration_in_days, category_art, category_music, category_crafts, category_comics, category_food, category_publishing, category_technology, category_journalism, category_design, category_film_and_video, category_theater, category_games, category_photography, category_fashion, country_GB, country_US, country_CA, country_other], index=['goal_usd', 'campaign_duration_in_days', 'category_art', 'category_music', 'category_crafts', 'category_comics', 'category_food', 'category_publishing', 'category_technology', 'category_journalism', 'category_design', 'category_film_and_video', 'category_theater', 'category_games', 'category_photography', 'category_fashion', 'country_GB', 'country_US', 'country_CA', 'country_other']) #this verbose index matches original df and serves generate_prediction() well

    return feature_vector


def generate_prediction(project_category, funding_goal, funding_period, country):
    """
    Accepts:
        'project_category': string
            --> must be from this list: 'art', 'music' 'crafts', 'comics', 'food', 'publishing', 'technology', 'journalism', 'design', 'film_and_video', 'theater', 'games', 'photography', 'fashion'
    
        'funding_goal':     int or float
    
        'funding_period':   int or float
    
        'country':          string
            --> must be from this list: 'GB', 'US', 'CA', 'other'

    Returns:
        prediction[0]:      numpy.int64
            --> 1==success, 0==failure
        
        pred_proba[0][1]:   numpy.float32
            --> probability of success
    """
    feature_vector = init_model_input() # reset-type function, defined above

    user_category = 'category_' + project_category
    user_country = 'country_' + country

    feature_vector['goal_usd'] = funding_goal 
    feature_vector['campaign_duration_in_days'] = funding_period
    feature_vector[user_category] = 1
    feature_vector[user_country] = 1

    X_user = feature_vector.to_numpy()
    X_user = X_user.reshape(1,-1)

    prediction = xgb_model.predict(X=X_user, validate_features=False)
    #example_output: array([1], dtype=int64)
    pred_proba = xgb_model.predict_proba(X=X_user, validate_features=False)
    #example output: array([[0.3698039, 0.6301961]], dtype=float32)
    
    return prediction[0], pred_proba[0][1]

