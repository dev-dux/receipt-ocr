from flask import Flask, render_template

app = Flask(__name__)

from app import routes

@app.route('/')
def index():
    return render_template('index.html')
