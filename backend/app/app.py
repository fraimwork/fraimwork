from flask import Flask, render_template, request
from flask_cors import CORS
from utils.gitutils import create_pull_request, clone_repo
from utils.agent import Agent
from utils.stringutils import arr_from_sep_string, extract_filename, extract_markdown_blocks
from utils.graphutils import build_file_tree_dag, string_represented_file_tree
import os, json, networkx as nx
from dotenv import load_dotenv
from utils.hive import Hive

app = Flask(__name__)
CORS(app)

load_dotenv()  # Load environment variables from .env

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
MODEL_NAME = os.getenv('MODEL_NAME')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
API_KEY = os.getenv('API_KEY')

translator = Agent(MODEL_NAME, api_key=API_KEY)
# translator.logged_prompt("You are a software engineer tasked with translating a framework.", asker="System")

architect = Agent(MODEL_NAME, api_key=API_KEY)
# architect.logged_prompt("You are a software engineer tasked with translating a framework. Your job is to describe the file tree structure that the new framework should use given the old one.", asker="System")

summarizer = Agent(MODEL_NAME, api_key=API_KEY)
# summarizer.logged_prompt("You are a high-level software engineer tasked with summarizing various code files.", asker="System")

hive = Hive([translator, architect, summarizer])

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
    for dir in os.listdir(repo_path):
        # if dir is a file
        if os.path.isfile(os.path.join(repo_path, dir)):
            os.remove(os.path.join(repo_path, dir))
        else:
            dir_name = extract_filename(dir)
            if dir_name == ".git" or dir_name in exceptions: continue
            # if dir is not empty, recursively wipe it
            if os.listdir(os.path.join(repo_path, dir)):
                wipe_repo(os.path.join(repo_path, dir))
            # os.rmdir(os.path.join(repo_path, dir))

def prepare_repo(repo_path, framework):
    working_dir = get_working_dir(framework)
    if not working_dir:
        return "Invalid framework"
    wipe_repo(repo_path)
    os.makedirs(f"{repo_path}/{working_dir}", exist_ok=True)
    return f'{repo_path}/{working_dir}'

def translate_code(source, target, code):
    prompt = f"Respond with code and nothing else. Translate the following {source} code to {target}: {code}"
    response = translator.prompt(prompt)
    return extract_markdown_blocks(response)[0]

def log_agents():
    translator.log_conversation('./logs/translator.log')
    architect.log_conversation('./logs/architect.log')
    summarizer.log_conversation('./logs/summarizer.log')

@app.route('/translate', methods=['POST'])
def translate():
    data = json.loads(request.data)
    repo_url = data['repo']
    source = data['source']
    target = data['target']

    if not repo_url:
        return "Error: Missing 'repo' parameter", 400

    # Clone the repo and make a new branch
    # if repo already exists, delete it
    path = f'./tmp/{repo_url.split("/")[-1].replace(".git", "")}'
    print(path)
    if os.path.exists(path):
        wipe_repo(path)
    repo = clone_repo(repo_url)
    base_branch = "master"
    created_branch = f"translation-{source}-{target}"
    repo.git.checkout(base_branch)
    repo.git.checkout('-b', created_branch)
    local_repo_path = str(repo.working_dir)
    working_dir_path = f'{local_repo_path}/{get_working_dir(source)}'

    # Get the file structure of the original repo and figure out which files to translate
    file_tree_structure = build_file_tree_dag(working_dir_path)
    # correspondance_graph = nx.DiGraph()
    # for node in file_tree_structure.nodes:
    #     response = architect.logged_prompt(f"Which file(s) from the old structure correspond to {node.path} in the new structure")
    #     arr_from_sep_string()
    #     correspondance_graph.add_edge(node)

    reverse_topology = list(nx.topological_sort(file_tree_structure))[::-1]
    files_to_translate = [node for node in reverse_topology if any(str(node).endswith(ext) for ext in extensions_of[source]) and node not in ignored_files_of[source]]
    print(f"Files to translate: {files_to_translate}")
    # working_dir_path = prepare_repo(local_repo_path, target)
    for file in files_to_translate:
        node = file_tree_structure.nodes[file]
        file_path = node['path']
        file_name = node['name']
        code = node['content']
        # Call Vertex Gemini API for translation (replace with actual API call)
        translated_code = translate_code(source, target, code)
        print(f"Translated code for {file_path}: {translated_code}")
        with open(f'{working_dir_path}/{file_path}', 'w') as f:
            f.write(translated_code)
    
    # Log the conversations with the agents
    log_agents()

    return create_pull_request(
        repo_link=repo_url,
        base_branch=base_branch,
        new_branch=created_branch,
        title=f"Translation from {source} to {target}",
        body=f"This is a boilerplate translation performed by the Fraimwork app. Please check to make sure that all logic is appropriately translated."
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
