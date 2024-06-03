from flask import Flask, render_template, request
import requests
from backend.utils.transformer import Transformer
from utils.returny import create_pull_request
from utils.training import download_training_data
import json
import os

app = Flask(__name__)

def success(x):
    return 200, x

def failure(x):
    return 400, x

# Get local .env file
from dotenv import load_dotenv
load_dotenv()



GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate_and_pr', methods=['POST'])
def translate_and_pr():
    repo_link = request.form['repo']
    base_branch = request.form['base_branch']
    source = request.form['source']
    target = request.form['target']
    # Fetch the repository
    response = requests.get(repo_link)
    if response.status_code != 200:
        return render_template('result.html', result="Failed to fetch repository")
    # Run the model and save the translated code to some path
    ...
    translated_code = "path/to/translated/code"
    created_branch = "..."
    # Create a pull request
    create_pull_request(
        repo_link=repo_link,
        base_branch=base_branch,
        new_branch=created_branch,
        title=f"Translation from {source.capitalize()} => {target.capitalize()}",
        body=f"This is a boilerplate translation performed by the Fraimwork app. Please check to make sure that all logic is appropriately translated."
    )
    return render_template('result.html', result="Success")

@app.route('/train', methods=['POST'])
def train_model():
    if 'source' not in request.form or 'target' not in request.form:
        return failure("Source and target frameworks must be specified.")
    source = request.form['source']
    target = request.form['target']
    if source == target:
        return failure("Source and target frameworks must be different.")
    # Check if model exists
    if f"models/{source}_to_{target}.pt" in os.listdir("models"):
        return render_template('result.html', message="Model already exists.")

    # Load training data
    download_training_data()

    # Generate vocabularies
    START_TOKEN = '<START>'
    PADDING_TOKEN = '<PAD>'
    END_TOKEN = '<END>'
    NEWLINE_TOKEN = '<NWLN>'
    END_OF_FILE_TOKEN = '<EOF>'
    CHANGE_DIR_TOKEN = '<CD|>'
    GENERICS = [START_TOKEN, PADDING_TOKEN, END_TOKEN, NEWLINE_TOKEN, END_OF_FILE_TOKEN]





if __name__ == '__main__':
    app.run(debug=True)
