import joblib
import numpy as np
import pandas as pd


xgb_model = joblib.load('xgb_model')


def init_model_input():

    """For prediction of funding success v. failure, app user may input feature values to model manually, via app HTML routes; but most feature values will remain as defined by this reset function."""
    
    goal_usd = 5000.                     # median = 5000

    campaign_duration_in_days = int(30)  # median = 30

    category_art, category_music, category_crafts, category_comics, category_food, category_publishing, category_technology, category_journalism, category_design, category_film_and_video, category_theater, category_games, category_photography, category_fashion, country_GB, country_US, country_CA, country_other = np.zeros(18, dtype=np.int64)

    feature_vector = pd.Series([goal_usd, campaign_duration_in_days, category_art, category_music, category_crafts, category_comics, category_food, category_publishing, category_technology, category_journalism, category_design, category_film_and_video, category_theater, category_games, category_photography, category_fashion, country_GB, country_US, country_CA, country_other], index=['goal_usd', 'campaign_duration_in_days', 'category_art', 'category_music', 'category_crafts', 'category_comics', 'category_food', 'category_publishing', 'category_technology', 'category_journalism', 'category_design', 'category_film_and_video', 'category_theater', 'category_games', 'category_photography', 'category_fashion', 'country_GB', 'country_US', 'country_CA', 'country_other'])

    return feature_vector


def generate_prediction(project_category, funding_goal, funding_period, country):
    """
    'goal': float
    'duration': int
    'category': string
        --> must be from this list: 'art', 'music' 'crafts', 'comics', 'food', 'publishing', 'technology', 'journalism', 'design', 'film_and_video', 'theater', 'games', 'photography', 'fashion'
    'country': string
        --> must be from this list: 'GB', 'US', 'CA', 'other'
    """
    feature_vector = init_model_input()

    user_category = 'category_' + project_category
    user_country = 'country_' + country

    feature_vector['goal_usd'] = funding_goal 
    feature_vector['campaign_duration_in_days'] = funding_period
    feature_vector[user_category] = int(1)
    feature_vector[user_country] = int(1)

    # feature_vector.tonumpy() # PROBABLY NOT NECESSARY; df ok (but pd.Series?)

    prediction = xgb_model.predict(X=feature_vector, validate_features=False)
    predict_proba = xgb_model.predict_proba(X=feature_vector, validate_features=False)
    
    return prediction, predict_proba

# <https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier>
# <https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.predict>
# <https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.predict_proba>
