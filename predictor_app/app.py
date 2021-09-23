'''Kickstarter_Success app logic'''

from flask import Flask, render_template, request


def create_app():
    """Create and configure an instance of the flask application"""
    app = Flask(__name__)

    # configure app
    # app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URI")
    # app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    with app.app_context():
        from .model_input_funcs import xgb_model, init_model_input, generate_prediction
        from .features_scrape import KickSoup


   # ROOT ROUTE
    @app.route('/', methods=["GET", "POST"])
    def root():     
        """Base view"""
        return render_template('main.html')
    

    @app.route('/foresight', methods=["GET", "POST"])
    def predict():

        if request.method == "POST":

            project_category = request.form["project_category"]
            funding_goal = request.form["funding_goal"]
            funding_period = request.form["funding_period"]
            country = request.form["country"]

            funding_goal = float(funding_goal)
            funding_period = float(funding_period)

            xgb_pred, xgb_proba = generate_prediction(
                project_category = project_category,
                funding_goal = funding_goal,
                funding_period = funding_period,
                country = country)

            if xgb_pred == 0:
                xgb_proba = 1-xgb_proba
                xgb_pred = 'failure'
            else:
                xgb_pred = 'SUCCESS!'

        return render_template(
            'reveal.html', xgb_pred=xgb_pred, xgb_proba=xgb_proba)


    # troubleshoot: hitting submit repeatedly is no good; we could fix that.


    # ... more routes? ... visuals? ...


    return app
