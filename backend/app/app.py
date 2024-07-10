from flask import Flask, render_template, request
from utils.returny import create_pull_request
import json, os, requests
from dotenv import load_dotenv
from google.cloud import aiplatform
from git import Repo

load_dotenv()  # Load environment variables from .env

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
MODEL_NAME = os.getenv('MODEL_NAME')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

languages_of = {
    "flutter": [".dart"],
    "react-native": [".ts, .js, .tsx, .jsx"]
    # Add more frameworks and their corresponding languages here
}

ignored_files_of = {
    "flutter": [],
    "react-native": []
    # Add more frameworks and their corresponding ignored files here
}

aiplatform.init(project=PROJECT_ID, location=LOCATION)

app = Flask(__name__)

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
    # 1. Prepare the prompt for Gemini
    prompt = f"Translate the following {source} code to {target}: {code}"

    # 2. Send the request to the Gemini API (replace with actual API call)
    response = gemini_api_call(prompt)

    # 3. Extract the translated code from the response
    translated_code = extract_translated_code(response)

    return translated_code

# Helper functions (to be implemented based on the Gemini API)
def gemini_api_call(prompt):
    # ... (Code to interact with the Gemini API)
    pass

def extract_translated_code(response):
    # ... (Code to parse the API response and extract the translated code)
    pass



@app.route('/translate', methods=['GET'])
def translate():
    repo_url = request.args.get('repo')
    source = request.args.get('source')
    target = request.args.get('target')
    base_branch = request.args.get('base_branch')
    created_branch = request.args.get('created_branch')

    if not repo_url:
        return "Error: Missing 'repo' parameter", 400
    # 1. Clone the repository
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    local_repo_path = f'./tmp/{repo_name}'
    Repo.clone_from(repo_url, local_repo_path)
    # 2. Identify relevant files for translation (based on framework detection)
    # ... (Logic to determine which files to translate)
    # 3. Translate files using Vertex Gemini
    for file_path in files_to_translate:
        with open(file_path, 'r') as f:
            code = f.read()
        # Call Vertex Gemini API for translation (replace with actual API call)
        translated_code = translate_code(source, target, code) 
        with open(file_path, 'w') as f:
            f.write(f.write(translated_code)
        # 4. Create a new repository and push translated code
        # ... (Logic to create a new repo and push changes)
        # 5. Create a pull request
        # ... (Logic to create a PR against the original repo)
        return "Translation and PR process initiated", 200

    create_pull_request(
        repo_link=repo_link,
        base_branch=base_branch,
        new_branch=created_branch,
        title=f"Translation from {source.capitalize()} => {target.capitalize()}",
        body=f"This is a boilerplate translation performed by the Fraimwork app. Please check to make sure that all logic is appropriately translated."
    )
    return render_template('result.html', result="Success")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
