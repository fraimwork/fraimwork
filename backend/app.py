from flask import Flask, render_template, request
from parsing.parser import dart_tokenizer, javascript_tokenizer
import requests
from machine_learning.transformer import Transformer
import torch
import json
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    text = request.form['text']
    source = 'flutter'
    target = 'react'
    # TODO: Translate Text
    translated_text = ""
    return render_template('result.html', translated_text=translated_text)

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

def get_repos(num_repos=10, idea="whatsapp"):
    """Fetches 10 Flutter and 10 React Native repos for a given `idea` from GitHub.

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
    data = get_repos()
    with open("data.json", "w") as file:
        json.dump(data, file)

def search_github_repos(query, framework, language, num_repos=5):
    url = f"https://api.github.com/search/repositories?q={query}+clone+{framework}+language:{language}&sort=stars&order=desc"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['items'][:num_repos]
    else:
        raise Exception("GitHub API request failed")

if __name__ == '__main__':
    app.run(debug=True)
