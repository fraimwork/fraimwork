from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from utils.gitutils import create_pull_request, clone_repo
from utils.agent import Agent, GenerationConfig, Interaction
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

    # Initialize the agent and team
    swe = Agent(
        model_name=MODEL_NAME,
        api_key=API_KEY,
        name="swe",
        generation_config=GenerationConfig(temperature=0.2),
        system_prompt="You are a software engineer tasked with translating code from one framework to another. Respond with code and nothing else."
    )

    pm = Agent(
        model_name=MODEL_NAME,
        api_key=API_KEY,
        name="pm",
        generation_config=GenerationConfig(max_output_tokens=4000),
        system_prompt="You are a high-level technical project manager tasked with the project of translating a framework. In the following prompts, you will be given instructions on what to do. Answer them to the best of your knowledge."
    )

    # Get the file structure of the original repo and the proposed file structure of the new repo
    source_file_tree = FileTree.from_directory(working_dir_path)
    reverse_level_order = source_file_tree.reverse_level_order()
    for node in tqdm(reverse_level_order):
        if 'content' not in source_file_tree.nodes[node]: continue
        summary = pm.chat(f'Summarize the functionality of the following code, be brief and to the point:\n{remove_indents(source_file_tree.nodes[node]["content"])}')
        source_file_tree.nodes[node]['summary'] = summary
    prompt = f"This is the file tree for the original {source} repo:\n{get_working_dir(source)}\\\n{source_file_tree}. Create a file tree for the new {target} repo in the new working directory {get_working_dir(target)}. Structure your response as follows:\n```\n{get_working_dir(target)}\\\nfile tree\n```"
    custom_context = [
            Interaction(
                prompt=f'Summary of {file}',
                response=source_file_tree.nodes[file]["summary"],
                ) for file in reverse_level_order if 'content' in source_file_tree.nodes[file]
            ]
    raw_tree = extract_markdown_blocks(pm.chat(prompt, custom_context=custom_context))[0][len(get_working_dir(target)) + 2:]
    wipe_repo(local_repo_path)
    working_dir_path = f'{local_repo_path}\\{get_working_dir(target)}'
    write_file_tree(raw_tree, working_dir_path)
    target_file_tree = FileTree.from_directory(working_dir_path)
    print(target_file_tree)

    # Create a correspondence graph between the two file trees
    correspondance_graph = nx.DiGraph()
    correspondance_graph.add_nodes_from(source_file_tree.nodes)
    ... # TODO
    return "wow"

    # Translate the code in the proposed file tree
    files_to_make = target_file_tree.reverse_level_order()
    for node in files_to_make:
        if 'content' in target_file_tree.nodes[node]: continue
        relevant_files = correspondance_graph[node]
        custom_context = [
            Interaction(
                prompt=f'{file}:\n{remove_indents(source_file_tree.nodes[file]["content"])}',
                response='Waiting for instructions to translate...',
            ) for file in relevant_files
        ]
        target_file_tree.nodes[node]['content'] = swe.chat(
            f"Translate the prior code from {source} to {target} to create {node}:",
            custom_context=custom_context
        )
        with open(f'{working_dir_path}\\{node}', 'w') as f:
            f.write(target_file_tree.nodes[node]['content'])

    # Create a pull request with the translated code
    return create_pull_request(
        repo_link=repo_url,
        base_branch=base_branch,
        new_branch=created_branch,
        title=f"Translation from {source} to {target}",
        body=f"This is a boilerplate translation performed by the Fraimwork app. Please check to make sure that all logic is appropriately translated."
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
