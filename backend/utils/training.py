import requests
import pandas as pd
import json

GITHUB_API_URL = 'https://api.github.com'

FRAMEWORK_LANGUAGE = {
    'django': 'python',
    'flask': 'python',
    'express': 'javascript',
    'react': 'javascript',
    'angular': 'javascript',
    'vue': 'javascript',
    'rails': 'ruby',
    'sinatra': 'ruby',
    'laravel': 'php',
    'symfony': 'php',
    'spring': 'java',
    'play': 'java',
    'flutter': 'dart',
    'react-native': 'javascript',
    'ionic': 'javascript',
    'cordova': 'javascript',
    'xamarin': 'csharp',
    'unity': 'csharp',
    'django-rest': 'python',
    'flask-rest': 'python',
    'express-rest': 'javascript',
    'rails-rest': 'ruby',
    'sinatra-rest': 'ruby',
}

def search_github_repos(idea, framework: str, num_repos=5):
    '''Search GitHub for repos with a certain idea'''
    url = f"{GITHUB_API_URL}/search/repositories?q={idea}+clone+{framework}+language:{FRAMEWORK_LANGUAGE[framework.lower()]}&sort=stars&order=desc"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['items'][:num_repos]
    else:
        raise Exception("GitHub API request failed")

def download_training_data(frameworkA, frameworkB):
    """Downloads training from GitHub and other sources and saves it to a directory."""
    dfA = pd.DataFrame()
    dfB = pd.DataFrame()
    with open('data/clones.txt', 'r') as file:
        clones = file.readlines()
    for clone in clones:
        reposA = search_github_repos(idea=clone, framework=frameworkA)
        reposB = search_github_repos(idea=clone, framework=frameworkB)
        for repo in reposA:
            dfA = dfA.add({
                'repo_name': repo['full_name'],
                'repo_url': repo['html_url'],
                'description': repo['description'],
                'stars': repo['stargazers_count'],
                'forks': repo['forks'],
                'language': repo['language'],
                'clone': clone,
                'framework': frameworkA
            }, ignore_index=True)
        for repo in reposB:
            dfB = dfB.add({
                'repo_name': repo['full_name'],
                'repo_url': repo['html_url'],
                'description': repo['description'],
                'stars': repo['stargazers_count'],
                'forks': repo['forks'],
                'language': repo['language'],
                'clone': clone,
                'framework': frameworkB
            }, ignore_index=True)