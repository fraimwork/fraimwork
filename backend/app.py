from flask import Flask, render_template, request
from parsing.parser import dart_tokenizer, javascript_tokenizer
import requests
from machine_learning.transformer import Transformer
import torch
import json
import numpy as np
import os

app = Flask(__name__)

# Get local .env file
from dotenv import load_dotenv
load_dotenv()



GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_API_URL = 'https://api.github.com'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate_and_pr', methods=['POST'])
def translate_and_pr():
    repo_link = request.form['repo']
    # Fetch the repository
    response = requests.get(repo_link)
    if response.status_code != 200:
        return render_template('result.html', result="Failed to fetch repository")
    # Run the model and save the translated code to some path
    ...
    translated_code = "path/to/translated/code"
    # Create a pull request

    return render_template('result.html', result="Success")

@app.route('/train', methods=['POST'])
def train_model(source: str, target: str):
    # Check if model exists
    if f"models/{source}_to_{target}.pt" in os.listdir("models"):
        return render_template('result.html', message="Model already exists.")

    # Load training data
    download_training_data()

    # Generate vocabularies
    START_TOKEN = '<START>'
    PADDING_TOKEN = '<PAD>'
    END_TOKEN = '<END>'
    VARIABLE_TOKENS = []
    # We allow up to 10000 variable tokens
    for i in range(10000):
        VARIABLE_TOKENS.append(f'<VAR_{i}>')

import requests
import os
import json

def create_pull_request(repo_link, base_branch, new_branch, title, body):
    # Obtain our GitHub personal access token
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

    # Construct the PR
    BASE_BRANCH = base_branch
    NEW_BRANCH = new_branch
    PR_TITLE = 'Translation by Fraimwork'
    PR_BODY = 'This pull request was created by Fraimwork to translate the repository from '

    # Set the headers for authorization
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Step 1: Get the SHA of the base branch (main)
    response = requests.get(
        f'https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/git/refs/heads/{BASE_BRANCH}',
        headers=headers
    )
    response.raise_for_status()
    base_sha = response.json()['object']['sha']

    # Step 2: Create a new branch
    data = {
        'ref': f'refs/heads/{NEW_BRANCH}',
        'sha': base_sha
    }
    response = requests.post(
        f'https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/git/refs',
        headers=headers,
        data=json.dumps(data)
    )
    response.raise_for_status()

    # Note: You would typically make some changes to the new branch here before committing and pushing

    # Step 3: Create a pull request
    data = {
        'title': PR_TITLE,
        'body': PR_BODY,
        'head': NEW_BRANCH,
        'base': BASE_BRANCH
    }
    response = requests.post(
        f'https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/pulls',
        headers=headers,
        data=json.dumps(data)
    )
    response.raise_for_status()
    pr_url = response.json()['html_url']

    print(f'Pull request created: {pr_url}')

def get_repos(num_repos=10, idea="whatsapp"):
    """Fetches `num_repos` Flutter and React Native repos for a given `idea` from GitHub.

    Args:
        num_repos (int, optional): Number of repositories to fetch for each framework. Defaults to 10.
        idea (str, optional): Idea to search for in repository descriptions. Defaults to "whatsapp".

    Returns:
        dict: Dictionary containing lists of Flutter and React Native repos.
    """
    # Base URL for GitHub search API
    base_url = "https://api.github.com/search/repositories"

    # Search queries for Flutter and React Native clones
    flutter_query = {
        "q": f"fork:public in:name {idea} (flutter OR dart)",
        "sort": "stars",
        "order": "desc",
        "per_page": num_repos
    }
    react_native_query = {
        "q": f"fork:public in:name {idea} react-native",
        "sort": "stars",
        "order": "desc",
        "per_page": num_repos
    }

    # Send requests for each framework
    flutter_response = requests.get(base_url, params=flutter_query)
    react_native_response = requests.get(base_url, params=react_native_query)

    # Check for successful responses
    if flutter_response.status_code == 200 and react_native_response.status_code == 200:
        # Parse JSON data
        flutter_data = json.loads(flutter_response.text)["items"]
        react_native_data = json.loads(react_native_response.text)["items"]

        # Extract repository information
        flutter_repos = [{ "name": repo["full_name"], "url": repo["html_url"] } for repo in flutter_data]
        react_native_repos = [{ "name": repo["full_name"], "url": repo["html_url"] } for repo in react_native_data]

        return {
            "flutter": flutter_repos,
            "react_native": react_native_repos
        }
    else:
        print(f"Error fetching data: {flutter_response.status_code} {react_native_response.status_code}")
        return {}

def download_training_data():
    """Downloads training from GitHub and saves it to a directory."""
    with open('data/clones.txt', 'r') as file:
        clones = file.readlines()
    data = [repo for repos in [get_repos(idea=clone) for clone in clones] for repo in repos]
    with open("data.json", "w") as file:
        json.dump(data, file)

def search_github_repos(query, framework, language, num_repos=5):
    url = f"{GITHUB_API_URL}/search/repositories?q={query}+clone+{framework}+language:{language}&sort=stars&order=desc"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['items'][:num_repos]
    else:
        raise Exception("GitHub API request failed")

if __name__ == '__main__':
    app.run(debug=True)
