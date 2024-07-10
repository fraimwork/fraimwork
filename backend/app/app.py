from flask import Flask, render_template, request
import requests
from utils.returny import create_pull_request
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

def get_working_dir(framework):
    match framework:
        case "flutter":
            return "./lib"
        case "react-native":
            return "./src"
        case _:
            return None

def translate_code(source, target, code):
    # Prompt google gemini
    pass

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
    # Clone the repository
    os.system(f"git clone {repo_link}")
    # Change directory
    os.chdir(repo_link.split('/')[-1].split('.')[0])
    reverse_level_order = os.walk(top=os.getcwd(), topdown=False)
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


if __name__ == '__main__':
    app.run(debug=True)
