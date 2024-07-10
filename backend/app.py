from flask import Flask, render_template, request
import requests
from utils.returny import create_pull_request
from utils.training import download_training_data
import json
import networkx as nx
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

def getFileDAG(path):
    G = nx.DiGraph()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".dart"):
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                    for i in range(len(lines)):
                        if i == 0:
                            G.add_edge(file, lines[i])
                        else:
                            G.add_edge(lines[i-1], lines[i])

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
    dag = getFileDAG(os.getcwd())
    reverse_level_order = list(reversed(nx.topological_sort(dag)))
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
