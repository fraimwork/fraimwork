from flask import Flask, render_template, request
from flask_cors import CORS
from utils.gitutils import create_pull_request, clone_repo
from utils.agent import Agent
from utils.stringutils import arr_from_sep_string, build_file_tree_dag, generate_tree_structure
import os, json, networkx as nx
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()  # Load environment variables from .env

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
MODEL_NAME = os.getenv('MODEL_NAME')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
API_KEY = os.getenv('API_KEY')

translator = Agent(MODEL_NAME, api_key=API_KEY)
# translator.logged_prompt("You are a software engineer tasked with translating a framework. Respond with a single block of code and nothing else.")

pm = Agent(MODEL_NAME, api_key=API_KEY)
# pm.logged_prompt("You are a project manager tasked with translating a framework. Your job is to describe the file tree structure that the new framework should use given the old one. Respond with a file tree structure in a markdown cell and nothing else.")

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
            return "lib"
        case "react-native":
            return "src"
        case _:
            return None

def wipe_repo(repo_path, exceptions=set()):
    # remove all files except for .git folder
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            if dir == ".git" or dir in exceptions: continue
            os.rmdir(os.path.join(root, dir))

def prepare_repo(repo_path, framework):
    working_dir = get_working_dir(framework)
    if not working_dir:
        return "Invalid framework"
    wipe_repo(repo_path)
    os.makedirs(f"{repo_path}/{working_dir}", exist_ok=True)
    return "Repo prepared"

def translate_code(source, target, code):
    prompt = f"Translate the following {source} code to {target}: {code}"
    response = translator.prompt(prompt)
    return extract_markdown(response)

def extract_markdown(text: str):
    # Split by  markdown ticks to get the code
    code = text.split("```")[1].split("```")[0]
    return code

@app.route('/translate', methods=['POST'])
def translate():
    data = json.loads(request.data)
    repo_url = data['repo']
    source = data['source']
    target = data['target']

    if not repo_url:
        return "Error: Missing 'repo' parameter", 400

    # Clone the repo and make a new branch
    repo = clone_repo(repo_url)
    base_branch = "master"
    created_branch = f"translation-{source}-{target}"
    repo.git.checkout(base_branch)
    repo.git.checkout('-b', created_branch)
    local_repo_path = str(repo.working_dir)
    working_dir_path = f'{local_repo_path}/{get_working_dir(source)}'

    # Get the file structure of the original repo and figure out which files to translate
    file_tree_structure = build_file_tree_dag(working_dir_path)
    reverse_topology = list(nx.topological_sort(file_tree_structure))[::-1]
    files_to_translate = [node for node in reverse_topology if any(str(node).endswith(ext) for ext in extensions_of[source]) and node not in ignored_files_of[source]]
    for file_path in files_to_translate:
        with open(f'{working_dir_path}/{file_path}', 'r') as f:
            code = f.read()
        # Call Vertex Gemini API for translation (replace with actual API call)
        translated_code = translate_code(source, target, code)
        print(f"Translated code for {file_path}: {translated_code}")
        with open(f'{working_dir_path}/{file_path}', 'w') as f:
            f.write(translated_code)

    return create_pull_request(
        repo_link=repo_url,
        base_branch=base_branch,
        new_branch=created_branch,
        title=f"Translation from {source} to {target}",
        body=f"This is a boilerplate translation performed by the Fraimwork app. Please check to make sure that all logic is appropriately translated."
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
