import requests
import json
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def hello_world():
   return 'Hello World'

if __name__ == '__main__':
   app.run()

# def connectToServer():
#     pass

# def getRequest():
#     pass

# def postRequest():
#     pass