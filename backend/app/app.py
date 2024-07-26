import math
import os, json, shutil, asyncio, networkx as nx
from flask import Flask, request
from flask_cors import CORS
from utils.gitutils import create_pull_request, clone_repo, create_branch
from utils.agent import Agent, GenerationConfig, Interaction, Team
from utils.stringutils import arr_from_sep_string, extract_markdown_blocks, markdown_to_dict
from utils.filetreeutils import FileTree, write_file_tree
from utils.languageutils import DartAnalyzer
from dotenv import load_dotenv
from utils.graphutils import loose_level_order, collapsed_level_order
import datetime
import time

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

def failed_check_for_params(data, *params):
    for param in params:
        if not data[param]:
            return f"Error: Missing '{param}' parameter", 400
    return None

@app.route('/translate', methods=['POST'])
async def translate():
    # Load the request data
    data = json.loads(request.data)
    if error := failed_check_for_params(data, 'repo', 'source', 'target'):
        return error 
    repo_url = data['repo']
    source = data['source']
    target = data['target']
    
    if target == source:
        return "Error: Source and target frameworks must be different", 400

    # Clone the repo and make a new branch
    repo = clone_repo(repo_url)
    base_branch = repo.active_branch.name
    created_branch = f"translation-{source}-{target}"
    create_branch(repo, repo.active_branch.name, created_branch)
    local_repo_path = str(repo.working_dir)
    working_dir_path = f'{local_repo_path}\\{get_working_dir(source)}'

    # Initialize the agent and team
    target_swe = Agent(
        model_name="gemini-1.5-flash-001",
        api_key=API_KEY,
        name="target_swe",
        generation_config=GenerationConfig(temperature=0.8),
        system_prompt=f"You are a software engineer tasked with writing {target} code based on a {source} repo. Answer the following prompts to the best of your skills."
    )
    pm = Agent(
        model_name="gemini-1.5-flash-001",
        api_key=API_KEY,
        name="pm",
        generation_config=GenerationConfig(temperature=0.5),
        system_prompt=f"You are a high-level technical project manager tasked with the project of translating a repository from {source} to {target}. Answer the following prompts to the best of your knowledge."
    )
    source_swe = Agent(
        model_name="gemini-1.5-flash-001",
        api_key=API_KEY,
        name="source_swe",
        generation_config=GenerationConfig(temperature=0.7),
        system_prompt=f'''You are a {source} software engineer. Your job is to summarize the functionality of provided code files. Structure your response as follows:

# filename.ext:

$7 sentence summary of the functionality/contents of the file$
    '''
    )
    team = Team(target_swe, pm, source_swe)

    analyzer = DartAnalyzer(working_dir_path)

    source_dependency_graph = analyzer.buildDependencyGraph()

    # loose level order
    eval_order = loose_level_order(source_dependency_graph)[::-1]

    (source_file_tree := FileTree.from_dir(working_dir_path))

    # Summarize the original repo
    async def summarize_group(group):
        async def summarize(node):
            name = source_file_tree.nodes[node]['name']
            content = source_file_tree.nodes[node]['content']
            message = f"{name}\n```\n{content}\n```"
            return await team.async_chat_with_agent(
                agent_name='source_swe',
                message=message,
                context_keys=[f"summary_{neighbor}" for neighbor in source_dependency_graph[node]],
                save_keys=[f"summary_{node}", "all"],
                prompt_title=f"Summary of {name}"
                )
        tasks = [summarize(node) for node in group]
        responses = await asyncio.gather(*tasks)
        for i, response in enumerate(responses):
            source_file_tree.nodes[group[i]]["summary"] = response

    for level in eval_order:
        await summarize_group(level)
        time.sleep(0.5)

    # Create new file tree
    prompt = f'''This is the file tree for the original {source} repo:
```
{get_working_dir(source)}\\
{source_file_tree}
```
You are tasked with re-structuring the directory to create a file tree for the new react-native repo in the new working directory {get_working_dir(target)}\\. Format your response as follows:
# Summary of Original Repo
$less than 10 sentence summary of the original repo$

# Methodology of Translation
$Brief explanation of what needs to be translated (include things like filename changes for entry points and other structural changes)$

# File Tree
```
{get_working_dir(target)}\\
├── folder1\\
│   ├── file1.ext
├── folder2\\
│   ├── folder3\\
│   │   ├── file2.ext
```
*Only include files of type(s) {extensions_of[target]} in your file tree

# All Correspondances
$List all the files in the original repo and their corresponding file(s) in the new repo (N/A if no such file(s) exists)$
```
{get_working_dir(source)}\\path\\to\\{source}\\file1.ext --> {get_working_dir(target)}\\path\\to\\{target}\\file1.ext, {get_working_dir(target)}\\...\\file2.ext
{get_working_dir(source)}\\path\\to\\{source}\\file2.ext --> {get_working_dir(target)}\\path\\to\\{target}\\file3.ext
{get_working_dir(source)}\\path\\to\\{source}\\file_with_no_correspondant.ext --> N/A
```
'''
    response = team.chat_with_agent('pm', prompt, context_keys=['all'], save_keys=['all'], prompt_title=f"Translate file tree from {source} to {target}:\n```\n{get_working_dir(source)}\\\n{source_file_tree}```")

    response_dict = markdown_to_dict(response)

    raw_tree = extract_markdown_blocks(response_dict['file tree'])[0][len(get_working_dir(target)) + 2:]

    correspondance_graph = nx.Graph()
    correspondences = arr_from_sep_string(extract_markdown_blocks(response_dict['all correspondances'])[0], '\n')
    for correspondence in correspondences:
        source_file, target_files = arr_from_sep_string(correspondence, ' --> ')
        source_file = source_file[len(get_working_dir(source)) + 1:]
        if not os.path.exists(f"{working_dir_path}\\{source_file}") or target_files == "N/A": continue
        target_files = arr_from_sep_string(target_files)
        for target_file in target_files:
            target_file = target_file[len(get_working_dir(target)) + 1:]
            correspondance_graph.add_edge(source_file, target_file)

    wipe_repo(local_repo_path)
    # repo.git.commit(A=True, m="Let the past die. Kill it if you have to")
    working_dir_path = f'{local_repo_path}\\{get_working_dir(target)}'
    write_file_tree(raw_tree, working_dir_path)
    # repo.git.commit(A=True, m="A New Hope")
    (target_file_tree := FileTree.from_dir(working_dir_path))

    # Cache the context threads
    factor = 32768 / pm.model.count_tokens('\n'.join(message['parts'][0] for interaction in team.context_threads['all'] for message in interaction.to_dict())).total_tokens
    # eg. if factor is 3.2, round to 4
    factor = math.ceil(factor)
    team.context_threads['all'] *= factor
    pm.model.count_tokens('\n'.join(message['parts'][0] for interaction in team.context_threads['all'] for message in interaction.to_dict())).total_tokens

    pm.cache_context(team.context_threads['all'], ttl=datetime.timedelta(minutes=5))

    order = [node for node in target_file_tree.nodes if 'content' in target_file_tree.nodes[node]]
    async def build_description(node):
        prompt = f'''For the file {node} in the new {target} repo. I want you to provide the following write-up:
$A brief description of what the file should contain (classes, functions, views etc.) in 20 lines or less$
'''
        return await team.async_chat_with_agent('pm', prompt, save_keys=[f"description_{node}"], prompt_title=f"Description of what to build for {node}")
    tasks = [build_description(node) for node in order]
    descriptions = await asyncio.gather(*tasks)
    for i, description in enumerate(descriptions):
        target_file_tree.nodes[order[i]]['description'] = description

    # Create a dependency graph for the target repo
    target_dependency_graph = nx.DiGraph()
    for level in eval_order:
        for node in level:
            if node not in correspondance_graph.nodes: continue
            # We add the correspondant node to the target dependency graph and connect it to the correspondant nodes of the source dependency graph
            target_nodes = list(correspondance_graph.neighbors(node))
            for target_node in target_nodes:
                target_dependency_graph.add_node(target_node)
                for source_node in source_dependency_graph.neighbors(node):
                    if source_node not in correspondance_graph.nodes: continue
                    source_node_correspondants = list(correspondance_graph.neighbors(source_node))
                    for source_node_correspondant in source_node_correspondants:
                        target_dependency_graph.add_edge(target_node, source_node_correspondant)

    # label each node in target dependency graph with a description
    for node in target_dependency_graph.nodes:
        target_dependency_graph.nodes[node]['description'] = ""
        if 'content' not in target_file_tree.nodes[node]: continue
        description = target_file_tree.nodes[node]['description']
        target_dependency_graph.nodes[node]['description'] = description

    eval_order = loose_level_order(target_dependency_graph, key='description')[::-1]

    # Translate the code in the proposed file tree
    async def make_level(group: list):
        async def make_file(node):
            if node not in target_file_tree.nodes: return None
            if node not in correspondance_graph.nodes: return None
            relevant_files = [file[len(source) + 2:] for file in correspondance_graph[node]]
            depends_on = [file for file in target_dependency_graph[node]]
            custom_context = [
                Interaction(
                    prompt=f'Contents of {file} in the {target} repo. This file is a dependency of {node} whose relative path is {os.path.relpath(file, node)}',
                    response=target_file_tree.nodes[file]["content"]
                ) for file in depends_on if file in target_file_tree.nodes and 'content' in target_file_tree.nodes[file]
            ] + [
                Interaction(
                    prompt=f'Contents of {file} in the {source} repo',
                    response=f'{file}:\n{source_file_tree.nodes[file]["content"]}',
                ) for file in relevant_files if file in source_file_tree.nodes and 'content' in source_file_tree.nodes[file]
            ] + team.context_threads[f'description_{node}']
            raw_resp = await target_swe.async_chat(
                f"Using the context from the prior {source} code and used {target} code, write code in {target} to create {node}. Respond with code and nothing else.",
                custom_context=custom_context
            )
            blocks = extract_markdown_blocks(raw_resp)
            if len(blocks) == 0: return None
            return blocks[0]
        tasks = [make_file(node) for node in group]
        responses = await asyncio.gather(*tasks)
        for i, response in enumerate(responses):
            if response is None: continue
            target_file_tree.nodes[group[i]]['content'] = response
            with open(f'{working_dir_path}\\{group[i]}', 'w') as f:
                f.write(response)
    for level in eval_order:
        await make_level(level)
        time.sleep(0.5)

    # Commit the translated code
    repo.git.commit(A=True, m=f"Boilerplate {target} code")
    repo.git.push('origin', created_branch, force=True)

    # Create a pull request with the translated code
    pr = create_pull_request(
            repo=repo,
            base_branch=base_branch,
            new_branch=created_branch,
            title=f"Translation from {source} to {target}",
            body=f"This is a boilerplate translation performed by the Fraimwork app. Please check to make sure that all logic is appropriately translated before merging.",
            token=GITHUB_TOKEN
        )

    shutil.rmtree('./tmp/', ignore_errors=True)
    return pr


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
