from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from utils.gitutils import create_pull_request, clone_repo
from utils.agent import Agent, GenerationConfig
from utils.stringutils import arr_from_sep_string, extract_markdown_blocks, remove_indents
from utils.filetreeutils import FileTree, write_file_tree
import os, json, networkx as nx
from dotenv import load_dotenv
from utils.team import Team
from tqdm import tqdm
import requests

app = Flask(__name__)
CORS(app)

load_dotenv()  # Load environment variables from .env

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
MODEL_NAME = os.getenv('MODEL_NAME')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
API_KEY = os.getenv('API_KEY')

translator = Agent(
    model_name=MODEL_NAME,
    api_key=API_KEY, name="translator",
    generation_config=GenerationConfig(temperature=0.2),
    system_prompt="You are a software engineer tasked with translating code from one language to another. Respond with code and nothing else."
)

architect = Agent(
    model_name=MODEL_NAME,
    api_key=API_KEY,
    name="architect",
    system_prompt="You are a software engineer tasked with translating a framework. Your job is to describe the file tree structure that the new framework should use given the old one."
)

summarizer = Agent(
    model_name=MODEL_NAME, api_key=API_KEY,
    name="summarizer",
    system_prompt="You are a high-level software engineer tasked with summarizing various code files. For each file, merely provide a summary of the file's contents as you see them. Keep your summaries brief and to the point."
)

hive = Team([translator, architect, summarizer])

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
    repo = request.form['repo']
    source = "flutter"  # Sample default value
    target = "react-native"  # Sample default value
    # Reroute to /translate with the repo URL and source and target frameworks
    response = requests.post('http://localhost:8080/translate', json={'repo': repo, 'source': source, 'target': target})
    return render_template('result.html', response=response.text, prompt=f"Translate {source} to {target} in the following repo: {repo}")

def get_working_dir(framework):
    match framework:
        case "flutter":
            return "lib"
        case "react-native":
            return "src"
        case _:
            return None

def wipe_repo(repo_path, exceptions=set()):
    for dir in os.listdir(repo_path):
        if os.path.isfile(os.path.join(repo_path, dir)):
            os.remove(os.path.join(repo_path, dir))
        else:
            dir_name = os.path.basename(dir)
            if dir_name == ".git" or dir_name in exceptions: continue
            if os.listdir(os.path.join(repo_path, dir)):
                wipe_repo(os.path.join(repo_path, dir))
            os.rmdir(os.path.join(repo_path, dir))

def prepare_repo(repo_path, framework):
    working_dir = get_working_dir(framework)
    if not working_dir:
        return "Invalid framework"
    wipe_repo(repo_path)
    os.makedirs(f"{repo_path}/{working_dir}", exist_ok=True)
    return f'{repo_path}/{working_dir}'

def translate_code(source, target, code):
    prompt = f"Respond with code and nothing else. Translate the following {source} code to {target}: {code}"
    response = translator.chat(prompt, context_key=None)
    return extract_markdown_blocks(response)[0]

@app.route('/translate', methods=['POST'])
def translate():
    data = json.loads(request.data)
    repo_url = data['repo']
    source = data['source']
    target = data['target']

    if not repo_url:
        return "Error: Missing 'repo' parameter", 400

    # Clone the repo and make a new branch
    if not os.path.exists('.\\tmp\\6165-MSET-CuttleFish\\TeamTrack'):
        repo = clone_repo(repo_url)
        base_branch = "master"
        created_branch = f"translation-{source}-{target}"
        repo.git.checkout(base_branch)
        repo.git.checkout('-b', created_branch)
        local_repo_path = str(repo.working_dir)
        working_dir_path = f'{local_repo_path}\\{get_working_dir(source)}'
    else:
        local_repo_path = '.\\tmp\\6165-MSET-CuttleFish\\TeamTrack'
        working_dir_path = '.\\tmp\\6165-MSET-CuttleFish\\TeamTrack\\lib'

    # Get the file structure of the original repo and the proposed file structure of the new repo
    file_tree = FileTree.from_directory(working_dir_path)
    # for node in tqdm(file_tree.reverse_level_order()):
    #     if 'content' in file_tree.nodes[node]:
    #         summary = summarizer.chat(remove_indents(file_tree.nodes[node]['content']), context_key=None)
    #         file_tree.nodes[node]['summary'] = summary
    prompt = f"This is the file tree for the original {source} repo:\n{get_working_dir(source)}\\\n{file_tree}. Create a file tree for the new {target} repo. Structure your response as follows:\n```\n{get_working_dir(target)}\\\nfile tree\n```"
    response = extract_markdown_blocks(architect.chat(prompt, context_key=None))[0][len(get_working_dir(target)) + 2:]
    wipe_repo(local_repo_path)

    write_file_tree(response, f'{local_repo_path}\\{get_working_dir(target)}')
    proposed_file_tree = FileTree.from_directory(f'{local_repo_path}\\{get_working_dir(target)}')
    print(proposed_file_tree)
    return "wow"
    correspondance_graph = nx.DiGraph()
    correspondance_graph.add_nodes_from(file_tree.nodes)

    node = 2
    # find neighbors of node
    # neighbors = correspondance_graph[]

    reverse_topology = list(nx.topological_sort(file_tree))[::-1]
    files_to_translate = [node for node in reverse_topology if any(str(node).endswith(ext) for ext in extensions_of[source]) and node not in ignored_files_of[source]]
    print(f"Files to translate: {files_to_translate}")
    # working_dir_path = prepare_repo(local_repo_path, target)
    # for file in files_to_translate:
    #     node = file_tree.nodes[file]
    #     file_path = node['path']
    #     file_name = node['name']
    #     code = node['content']
    #     # Call Vertex Gemini API for translation (replace with actual API call)
    #     translated_code = translate_code(source, target, remove_indents(code))
    #     print(f"Translated code for {file_path}: {translated_code}")
    #     with open(f'{working_dir_path}/{file_path}', 'w') as f:
    #         f.write(translated_code)

    return create_pull_request(
        repo_link=repo_url,
        base_branch=base_branch,
        new_branch=created_branch,
        title=f"Translation from {source} to {target}",
        body=f"This is a boilerplate translation performed by the Fraimwork app. Please check to make sure that all logic is appropriately translated."
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
