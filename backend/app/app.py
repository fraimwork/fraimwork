from flask import Flask, render_template, request
from utils.gitutils import create_pull_request
import json, os, requests
from dotenv import load_dotenv
from git import Repo
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
MODEL_NAME = os.getenv('MODEL_NAME')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
API_KEY = os.getenv('API_KEY')

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

extensions_of = {
    "flutter": [".dart"],
    "react-native": [".ts, .js, .tsx, .jsx"]
    # Add more frameworks and their corresponding languages here
}

ignored_files_of = {
    "flutter": [],
    "react-native": []
    # Add more frameworks and their corresponding ignored files here
}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    code = request.form['code']
    source = "flutter"  # Replace with actual framework detection logic
    target = "react-native"  # Replace with actual framework detection logic
    translated_code = translate_code(source, target, code)
    return render_template('result.html', result=translated_code, prompt=code)

def get_working_dir(framework):
    match framework:
        case "flutter":
            return "./lib"
        case "react-native":
            return "./src"
        case _:
            return None

def wipe_repo(repo_path):
    # remove all files except for .git folder
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            if dir == ".git": continue
            os.rmdir(os.path.join(root, dir))

def prepare_repo(repo_path, framework):
    working_dir = get_working_dir(framework)
    if not working_dir:
        return "Invalid framework"
    wipe_repo(repo_path)
    os.makedirs(f"{repo_path}/{working_dir}", exist_ok=True)
    return "Repo prepared"

def translate_code(source, target, code):
    # Prepare the prompt for Gemini
    prompt = f"Translate the following {source} code to {target}: {code}"
    # Send the request to the Gemini API (replace with actual API call)
    response = gemini_api_call(prompt)
    # Extract the translated code from the response
    translated_code = extract_translated_code(response)
    return translated_code

# Helper functions (to be implemented based on the Gemini API)
def gemini_api_call(prompt):
    return model.generate_content(prompt).text

def extract_translated_code(response):
    text = response
    # Split by  markdown ticks to get the code
    code = text.split("```")[1].split("```")[0]
    return code

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
            f.write(translated_code)
        # 4. Create a new repository and push translated code
        # ... (Logic to create a new repo and push changes)
        # 5. Create a pull request
        # ... (Logic to create a PR against the original repo)
        return "Translation and PR process initiated", 200

    create_pull_request(
        repo_link=repo_url,
        base_branch=base_branch,
        new_branch=created_branch,
        title=f"Translation from {source.capitalize()} to {target.capitalize()}",
        body=f"This is a boilerplate translation performed by the Fraimwork app. Please check to make sure that all logic is appropriately translated."
    )
    return render_template('result.html', result="Success")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
