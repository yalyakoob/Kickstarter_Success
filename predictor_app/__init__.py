'''Kickstarter_Success app entry-point'''

from .app import create_app
# from dotenv import load_dotenv


# load_dotenv('.env')

APP = create_app()