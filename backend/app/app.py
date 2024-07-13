import requests
import asyncio
from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from utils.gitutils import create_pull_request, clone_repo, create_branch
from utils.agent import Agent, GenerationConfig, Interaction
from utils.stringutils import arr_from_sep_string, extract_markdown_blocks, remove_indents
from utils.filetreeutils import FileTree, write_file_tree
import os, json, networkx as nx
from dotenv import load_dotenv
from utils.team import Team
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
async def translate():
    data = json.loads(request.data)
    repo_url = data['repo']
    source = data['source']
    target = data['target']

    if not repo_url:
        return "Error: Missing 'repo' parameter", 400

    # Clone the repo and make a new branch
    repo = clone_repo(repo_url)
    base_branch = repo.active_branch.name
    created_branch = f"translation-{source}-{target}"
    create_branch(repo, repo.active_branch.name, created_branch)
    local_repo_path = str(repo.working_dir)
    working_dir_path = f'{local_repo_path}\\{get_working_dir(source)}'

    # Initialize the agent and team
    swe = Agent(
        model_name=MODEL_NAME,
        api_key=API_KEY,
        name="swe",
        generation_config=GenerationConfig(temperature=0.9),
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
    source_reverse_level_order = source_file_tree.reverse_level_order()
    async def task(node):
        if 'content' not in source_file_tree.nodes[node]: return None
        return await pm.async_chat(f'Summarize the functionality of the following code, be brief and to the point:\n{remove_indents(source_file_tree.nodes[node]["content"])}')
    tasks = [task(node) for node in source_reverse_level_order]
    responses = await asyncio.gather(*tasks)
    for i, response in enumerate(responses):
        source_file_tree.nodes[source_reverse_level_order[i]]["summary"] = response
    prompt = f"This is the file tree for the original {source} repo:\n{get_working_dir(source)}\\\n{source_file_tree}. Create a file tree for the new {target} repo in the new working directory {get_working_dir(target)}. Structure your response as follows:\n```\n{get_working_dir(target)}\\\nfile tree\n```"
    custom_context = [
            Interaction(
                prompt=f'Summary of {file}',
                response=source_file_tree.nodes[file]["summary"],
                ) for file in source_reverse_level_order if 'content' in source_file_tree.nodes[file]
            ]

    raw_tree = extract_markdown_blocks(pm.chat(prompt, custom_context=custom_context))[0][len(get_working_dir(target)) + 2:]
    wipe_repo(local_repo_path)
    working_dir_path = f'{local_repo_path}\\{get_working_dir(target)}'
    write_file_tree(raw_tree, working_dir_path)
    target_file_tree = FileTree.from_directory(working_dir_path)

    # Create a correspondence graph between the two file trees
    correspondance_graph = nx.DiGraph()
    custom_context.append(
            Interaction(
                prompt=prompt,
                response=raw_tree
            )
        )

    async def find_correspondance(node):
        if 'content' not in target_file_tree.nodes[node]: return None
        raw_correspondances = await pm.async_chat(f"Which file(s) in the original {source} repo correspond to {node} in the translated {target} repo you made? Only include the paths in your response and format your response as:\n```\n{get_working_dir(source)}\\path\\to\\{source}\\file1, {get_working_dir(source)}\\path\\to\\{source}\\file2\n```", custom_context=custom_context)
        return extract_markdown_blocks(raw_correspondances)[0]
    tasks = [find_correspondance(node) for node in target_file_tree.nodes]
    responses = await asyncio.gather(*tasks)
    for i, response in enumerate(responses):
        if not response: continue
        node = list(target_file_tree.nodes)[i]
        target_node = f'{target}_{node}'
        for correspondance in arr_from_sep_string(response):
            source_node = f'{source}_{correspondance[len(get_working_dir(source)) + 1:]}'
            correspondance_graph.add_edge(target_node, source_node)

    # Translate the code in the proposed file tree
    files_to_make = target_file_tree.reverse_level_order()
    async def make_file(node):
        if 'content' not in target_file_tree.nodes[node]: return None
        relevant_files = [file[len(target) + 2:] for file in correspondance_graph[f'{target}_{node}']]
        actually_relevant_files = []
        for source_file in source_file_tree.nodes:
            for file in relevant_files:
                if file in source_file:
                    actually_relevant_files.append(source_file)
        custom_context = [
            Interaction(
                prompt=f'{file}:\n{remove_indents(source_file_tree.nodes[file]["content"])}',
                response='Waiting for instructions to translate...',
            ) for file in actually_relevant_files
        ]
        raw_resp = await swe.async_chat(
            f"Translate the prior code from {source} to {target} to create {node}:",
            custom_context=custom_context
        )
        blocks = extract_markdown_blocks(raw_resp)
        if len(blocks) == 0: return "UNABLE TO TRANSLATE THIS FILE"
        return blocks[0]
    tasks = [make_file(node) for node in files_to_make]
    responses = await asyncio.gather(*tasks)
    for i, response in enumerate(responses):
        if not response: continue
        target_file_tree.nodes[files_to_make[i]]['content'] = response
        with open(f'{working_dir_path}\\{files_to_make[i]}', 'w') as f:
            f.write(response)

    # Commit the translated code
    repo.git.add(A=True)
    repo.index.commit(f"Translate code from {source} to {target}")
    # Push the branch to the remote repository
    repo.git.push("origin", created_branch)

    # Create a pull request with the translated code
    return create_pull_request(
            repo=repo,
            base_branch=base_branch,
            new_branch=created_branch,
            title=f"Translation from {source} to {target}",
            body=f"This is a boilerplate translation performed by the Fraimwork app. Please check to make sure that all logic is appropriately translated.",
            token=GITHUB_TOKEN
        )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
