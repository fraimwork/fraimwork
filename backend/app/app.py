import datetime, time, math, stat, os, json, shutil, asyncio, re, networkx as nx
from flask import Flask, request
from flask_cors import CORS
from utils.gitutils import create_pull_request, clone_repo, create_branch, wipe_repo, prepare_repo
from utils.agent import Agent, GenerationConfig, Interaction, Team
from utils.stringutils import arr_from_sep_string, extract_markdown_blocks, markdown_to_dict
from utils.filetreeutils import FileTree, write_file_tree
from utils.frameworkutils import DartAnalyzer
from dotenv import load_dotenv
from utils.graphutils import loose_level_order, collapsed_level_order
from utils.frameworkutils import Framework
from pathlib import Path
from utils.languageutils import get_imports
from utils.listutils import flatten, compact, singleton
from utils.errorutils import repeat_until_finish
from utils.firebase.firestoreutils import create_document, read_document, update_document, delete_document, write_graph_to_firestore

app = Flask(__name__)
CORS(app)

load_dotenv()  # Load environment variables from .env

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
MODEL_NAME = os.getenv('MODEL_NAME')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
API_KEY = os.getenv('API_KEY')

def failed_check_for_params(data, *params):
    for param in params:
        if not data[param]:
            return f"Error: Missing '{param}' parameter", 400
    return None

def clear_cache():
    def readonly_to_writable(foo, file, err):
        if Path(file).suffix in ['.idx', '.pack'] and 'PermissionError' == err[0].__name__:
            os.chmod(file, stat.S_IWRITE)
            foo(file)
    return shutil.rmtree('./tmp/', onerror=readonly_to_writable)

def classify_repo(local_repo_path):
    (file_tree := FileTree.from_dir(local_repo_path))
    frameworks = ', '.join(Framework.get_frameworks())
    classifier = Agent(
        model_name="gemini-1.5-flash-001",
        api_key=API_KEY,
        name="classifier",
        generation_config=GenerationConfig(temperature=0.3),
        system_prompt=f"""You are a software engineer tasked with classifying the framework of a codebase into one of the  following frameworks:
    {frameworks}.
    -----------------------------------
    In the following prompts, you will be given the subdirectories of a codebase. Directory names can be misleading. You can respond with one of the following:
    - IDENTIFY: FRAMEWORK_NAME (Only select this if you are absolutely certain that the ROOT directory conforms) (it should be used sparingly)
    - N/A (if you cannot identify the framework based on the subdirectories)
    - END (if you are fairly certain that no frameworks are in the subdirectories)"""
    )
    # RegEx to match classifier responses
    identify_re = re.compile(r"IDENTIFY: (.+)")
    end_re = re.compile(r"END")
    unable_re = re.compile(r"N/A")

    def classify_node(node):
        node_dict = file_tree.nodes[node]
        neighbors = list(file_tree[node])
        prompt = f"{file_tree.nodes[node]['name']}\n" + f"{file_tree.subfiletree(node).copy(withDepth=1)}"
        context = [Interaction(f'Relevant content of {neighbor}\n----\n' + '\n'.join(imports), '...') for  neighbor in  neighbors if 'content' in file_tree.nodes[neighbor] and (imports := get_imports(file_tree.nodes[neighbor]['path']))]
        response = classifier.chat(prompt, custom_context=context)
        # Match the response to the RegEx
        if (m := identify_re.match(response)):
            framework  = m.group(1).strip()
            print(f"{node_dict['name']} is a {framework} project")
            file_tree.nodes[node]['frameworks'] = [framework]
            for node in nx.dfs_tree(file_tree, node).nodes:
                file_tree.nodes[node]['frameworks'] = [framework]
            return [framework]
        elif (m := unable_re.match(response)):
            print(f"Searching {node} ...")
            frameworks = compact(flatten([classify_node(node) for node in neighbors if 'content' not in file_tree.nodes[node]]))
            file_tree.nodes[node]['frameworks'] = frameworks
        elif (m := end_re.match(response)):
            print(f"{node_dict['name']} is not a framework")
            return None
        else:
            print(f"Classifier response: {response}")
            return None
    
    return (root_classification := classify_node(file_tree.root_node()))


@app.route('/analyze', methods=['POST'])
async def analyze():
    data = json.loads(request.data)
    if error := failed_check_for_params(data, 'repo'):
        return error
    repo_url = data['repo']
    # Clone the repo
    repo = clone_repo(repo_url)
    local_repo_path = str(repo.working_dir)
    res = classify_repo(local_repo_path)
    clear_cache()
    return res


@app.route('/translate', methods=['POST'])
async def translate():
    # Load the request data
    data = json.loads(request.data)
    if error := failed_check_for_params(data, 'repo', 'extended_path', 'target'):
        return error 
    repo_url = data['repo']
    source = Framework[data['source']]
    target = Framework[data['target']]
    
    if target == source:
        return "Error: Source and target frameworks must be different", 400

    # Clone the repo and make a new branch
    repo = clone_repo(repo_url)
    base_branch = repo.active_branch.name
    created_branch = f"translation-{source}-{target}"
    create_branch(repo, repo.active_branch.name, created_branch)
    local_repo_path = str(repo.working_dir)
    working_dir_path = f'{local_repo_path}\\{source.get_working_dir()}'

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

    analyzer = source.get_analyzer(working_dir_path)

    source_dependency_graph = analyzer.buildDependencyGraph()

    # loose level order
    eval_order = loose_level_order(source_dependency_graph)[::-1]
    for level in eval_order:
        print(level)
    print(f"Number of levels: {len(eval_order)}")

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
                prompt_title=f"Summary of {node}"
                )
        tasks = [summarize(node) for node in group]
        responses = await asyncio.gather(*tasks)
        for i, response in enumerate(responses):
            source_file_tree.nodes[group[i]]["summary"] = response

    for level in eval_order:
        await summarize_group(level)
        time.sleep(0.5)

    curr_string = '\n'.join([message['parts'][0] for interaction in team.context_threads['all'] for message in interaction.to_dict()])
    (curr_tokens := pm.model.count_tokens(curr_string).total_tokens)

    would_be_string = '\n'.join([source_file_tree.nodes[node]['content'] for node in source_dependency_graph.nodes])
    (would_be_tokens := pm.model.count_tokens(would_be_string).total_tokens)

    (percent_tokens_reduction := 1 - (curr_tokens / would_be_tokens))

    # Create new file tree
    prompt = f'''This is the file tree for the original {source} repo:
```
{source.get_working_dir()}\\
{source_file_tree}
```
You are tasked with re-structuring the directory to create a file tree for the new react-native repo in the new working directory {target.get_working_dir()}\\. Format your response as follows:
# Summary of Original Repo
$less than 10 sentence summary of the original repo$

# Methodology of Translation
$Brief explanation of what needs to be translated (include things like filename changes for entry points and other structural changes)$

# File Tree
```
{target.get_working_dir()}\\
├── folder1\\
│   ├── file1.ext
├── folder2\\
│   ├── folder3\\
│   │   ├── file2.ext
```
*Only include files of type(s) {target.get_file_extensions()} in your file tree

# All Correspondances
$List all the files in the original repo and their corresponding file(s) in the new repo (N/A if no such file(s) exists)$
```
{source.get_working_dir()}\\path\\to\\{source}\\file1.ext --> {target.get_working_dir()}\\path\\to\\{target}\\file1.ext, {target.get_working_dir()}\\path\\to\\{target}\\file2.ext
{source.get_working_dir()}\\path\\to\\{source}\\file2.ext --> {target.get_working_dir()}\\path\\to\\{target}\\file3.ext
{source.get_working_dir()}\\path\\to\\{source}\\file_with_no_correspondant.ext --> N/A *do NOT provide a reason if n/a*
N/A *do NOT provide a reason if n/a* --> {target.get_working_dir()}\\path\\to\\{target}\\file_with_no_correspondant.ext
```
'''
    response = team.chat_with_agent('pm', prompt, context_keys=['all'], save_keys=['all'], prompt_title=f"Translate file tree from {source} to {target}:\n```\n{source.get_working_dir()}\\\n{source_file_tree}```")

    response_dict = markdown_to_dict(response)

    raw_tree = extract_markdown_blocks(response_dict['file tree'])[0][len(target.get_working_dir()) + 2:]

    wipe_repo(local_repo_path)
    # repo.git.commit(A=True, m="rm past")
    working_dir_path = f'{local_repo_path}\\{target.get_working_dir()}'
    write_file_tree(raw_tree, working_dir_path)
    # repo.git.commit(A=True, m="A New Hope")
    (target_file_tree := FileTree.from_dir(working_dir_path))

    correspondance_graph = nx.Graph()
    correspondences = arr_from_sep_string(extract_markdown_blocks(response_dict['all correspondances'])[0], '\n')
    for correspondence in correspondences:
        source_file, target_files = arr_from_sep_string(correspondence, ' --> ')
        if target_files == "N/A": continue
        source_file = source_file[len(source.get_working_dir()) + 1:]
        if not source_file_tree.has_node(source_file): continue
        target_files = arr_from_sep_string(target_files)
        for target_file in target_files:
            target_file = target_file[len(target.get_working_dir()) + 1:]
            if not target_file_tree.has_node(target_file): continue
            correspondance_graph.add_edge(source_file, target_file)

    # print edges
    for edge in correspondance_graph.edges:
        print(edge)

    pm.model.count_tokens('\n'.join(message['parts'][0] for interaction in team.context_threads['all'] for message in interaction.to_dict())).total_tokens
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
$Example usage$
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
            target_nodes = list(correspondance_graph[node])
            for target_node in target_nodes:
                target_node = target_node
                target_dependency_graph.add_node(target_node)
                for source_node in source_dependency_graph[node]:
                    if source_node not in correspondance_graph.nodes: continue
                    source_node_correspondants = list(correspondance_graph[source_node])
                    for source_node_correspondant in source_node_correspondants:
                        target_dependency_graph.add_edge(target_node, source_node_correspondant)

    # label each node in target dependency graph with a description
    for node in target_dependency_graph.nodes:
        target_dependency_graph.nodes[node]['description'] = ""
        if node not in target_file_tree.nodes:
            # maybe remove the node from the target dependency graph
            continue
        if 'content' not in target_file_tree.nodes[node]: continue
        description = target_file_tree.nodes[node]['description']
        target_dependency_graph.nodes[node]['description'] = description

    target_eval_order = loose_level_order(target_dependency_graph, key='description')[::-1]
    for level in target_eval_order:
        print(level)
    print(f"Number of levels: {len(target_eval_order)}")

    async def make_file(node):
        if node not in target_file_tree.nodes: return None
        if node not in correspondance_graph.nodes:
            print(node)
            return None
        relevant_files = list(correspondance_graph[node])
        depends_on = list(target_dependency_graph[node])
        dependencies = '\n'.join(os.path.relpath(target_file_tree.nodes[file]['path'], os.path.dirname(target_file_tree.nodes[node]['path'])) for file in depends_on)
        custom_context = [
            Interaction(
                prompt=f'Description of {file} in the {target} repo. This file is a dependency of {node}.',
                response=target_file_tree.nodes[file]["description"]
            ) for file in depends_on if file in target_file_tree.nodes and 'content' in target_file_tree.nodes[file]
        ] + [
            Interaction(
                prompt=f'Contents of {file} in the {source} repo',
                response=f'{file}:\n{source_file_tree.nodes[file]["content"]}',
            ) for file in relevant_files if file in source_file_tree.nodes and 'content' in source_file_tree.nodes[file]
        ] + team.context_threads[f'description_{node}']
        raw_resp = await target_swe.async_chat(
            f"The relative dependencies for {node} are\n{dependencies}. Using the context from the prior {source} code (to be translated) and {target} code which {node} depends on, write code in {target} to create {node}. Respond with code and nothing else.",
            custom_context=custom_context
        )
        blocks = extract_markdown_blocks(raw_resp)
        if len(blocks) == 0: return None
        return blocks[0]
    nodes = [node for level in target_eval_order for node in level]
    tasks = [make_file(node) for node in nodes]
    responses = await asyncio.gather(*tasks)
    for i, response in enumerate(responses):
        if response is None: continue
        target_file_tree.nodes[nodes[i]]['content'] = response
        with open(f'{working_dir_path}\\{nodes[i]}', 'w') as f:
            f.write(response)

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
    clear_cache()
    return pr

@app.route('/modify', methods=['POST'])
async def modify():
    data = json.loads(request.data)
    if error := failed_check_for_params(data, 'repo'):
        return error
    repo_url = data['repo']
    feature = f"""# Name: Organizations
    # Description: Organizations are a way to group users together. They can be used to automatically share events with the members of the organization whenever they are uploaded to the cloud. \
    Organizations consist of a name, team number, and a backend-generated unique id. They can also have a profile picture, description, and location. \
    Members can also opt to be publicly listed as a member of the organization or not. \
    """

    # Clone the repo and make a new branch
    repo = clone_repo(repo_url)
    base_branch = repo.active_branch.name
    created_branch = f"modification-{base_branch}"
    local_repo_path = str(repo.working_dir)
    create_branch(repo, base_branch, created_branch)
    source = singleton(classify_repo(local_repo_path))
    if not source:
        return "Error: Could not classify the repo", 400
    working_dir_path = f'{local_repo_path}\\{source.get_working_dir()}'

    source_swe_summarizer = Agent(
        model_name="gemini-1.5-flash-001",
        api_key=API_KEY,
        name="source_swe_analyzer",
        generation_config=GenerationConfig(temperature=0.7),
        system_prompt=f'''You are a {source} software engineer. Your job is to summarize the functionality of provided code files. Structure your response as follows:
# filename.ext:
$7 sentence summary of the functionality/contents of the file$
'''
    )
    pm = Agent(
        model_name="gemini-1.5-pro-001",
        api_key=API_KEY,
        name="pm",
        generation_config=GenerationConfig(temperature=0.3),
        system_prompt=f"""You are a high-level technical project manager tasked with the project of adding functionality to a {source} repository."""
    )
    source_swe_coder = Agent(
        model_name="gemini-1.5-flash-001",
        api_key=API_KEY,
        name="source_swe_coder",
        generation_config=GenerationConfig(temperature=0.7),
        system_prompt=f"You are a {source} software engineer. Your job is to write code to implement a specific feature. Respond with code and nothing else. No explanations needed."
    )
    team = Team(source_swe_summarizer, pm, source_swe_coder)
    analyzer = source.get_analyzer(working_dir_path)
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
                agent_name='source_swe_analyzer',
                message=message,
                context_keys=[f"summary_{neighbor}" for neighbor in source_dependency_graph[node]],
                save_keys=[f"summary_{node}", "all"],
                prompt_title=f"Summary of {node}"
                )
        tasks = [summarize(node) for node in group]
        responses = await asyncio.gather(*tasks)
        for i, response in enumerate(responses):
            source_file_tree.nodes[group[i]]["summary"] = response

    for level in tqdm(eval_order):
        await summarize_group(level)
        time.sleep(0.5)

    prompt = f"""This is the file tree for the original {source} repo:
    ```
    {source.get_working_dir()}\\
    {source_file_tree}
    ```
    Given the prior context, summarize the functionality of the entire repository succinctly.
    """
    print(team.chat_with_agent('source_swe_analyzer', prompt, context_keys=["all"], save_keys=["all"], prompt_title=f"""Repository Summary.
    File Tree provided:
    ```
    {source.get_working_dir()}\\
    {source_file_tree}
    ```
    """))

    curr_string = '\n'.join([message['parts'][0] for interaction in team.context_threads['all'] for message in interaction.to_dict()])
    (curr_tokens := pm.model.count_tokens(curr_string).total_tokens)

    would_be_string = '\n'.join([source_file_tree.nodes[node]['content'] for node in source_dependency_graph.nodes])
    (would_be_tokens := pm.model.count_tokens(would_be_string).total_tokens)

    (percent_tokens_reduction := 1 - (curr_tokens / would_be_tokens))

    response_format = f"""
You may choose ACTIONS from the following list:
- WRITE | $FILE_PATH$ | $PROMPT$ (PROMPT should inform how the new FILE_PATH should be written)
- EDIT | $FILE_PATH$ | $PROMPT$ (Edit the provided file to fulfill the prompt)
- DELETE | $FILE_PATH$ (Delete the provided file)
----------
Here is a sample response:
```
1. WRITE | {source.get_working_dir()}\\path\\to\\newfile.ext | Implement ... with the following properties:
    - property1
    - property2
2. EDIT | {source.get_working_dir()}\\path\\to\\file.ext | Fix the bug ... by doing:
    - step1
    - step2
3. DELETE | {source.get_working_dir()}\\path\\to\\some\\file.ext | Reasons for deletion:
    - reason1
    - reason2
```
"""

    # Create new file tree
    prompt = f'''
    You are to build the following feature:
    {feature}
    ------------------
    {response_format}
    ------------------
    You need to follow the response template structure exactly or the response will not be accepted.
    '''
    print(response := team.chat_with_agent("pm", prompt, context_keys=["all"], save_keys=["all"], prompt_title="Feature Creation Steps"))

    pattern  = re.compile(r'^\d+\.\s*((?:.*\n?)*?(?=\n\d+\.|\Z))', re.MULTILINE)
    md = extract_markdown_blocks(response)[0]
    lines = pattern.findall(md)
    actions = [(action[0].strip(), action[1].strip()[len(source.get_working_dir()) + 1:], action[2].strip()) for line in lines if (action := line.split('|'))]
    for action in actions:
        print(action)
    source_swe_coder.model.count_tokens('\n'.join(message['parts'][0] for interaction in team.context_threads['all'] for message in interaction.to_dict())).total_tokens

    # Cache the context threads
    factor = 32768 / source_swe_coder.model.count_tokens('\n'.join(message['parts'][0] for interaction in team.context_threads['all'] for message in interaction.to_dict())).total_tokens
    # eg. if factor is 3.2, round to 4
    factor = math.ceil(factor)
    thread = team.context_threads['all'].copy()
    thread *= factor
    source_swe_coder.model.count_tokens('\n'.join(message['parts'][0] for interaction in thread for message in interaction.to_dict())).total_tokens

    source_swe_coder.cache_context(thread, ttl=datetime.timedelta(minutes=8))

    def write(file_path, prompt):
        response = source_swe_coder.chat(prompt)
        md = extract_markdown_blocks(response)[0]
        path = f"{working_dir_path}\\{file_path}"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            f.write(md)

    def edit(file_path, prompt):
        with open(f"{working_dir_path}\\{file_path}", 'r') as f:
            content = f.read()
        response = source_swe_coder.chat(f"""{prompt}
Respond in the following format:
# Old Code Snippet
```
$old code snippet to be changed (this string must match exactly with a substring of the original content) (no filler comments etc.)$
```
# New Code Snippet
```
$the code snippet the prompt is asking for$
```""", custom_context=[Interaction(f"Contents of {file_path}", content)])
        old_code, new_code = extract_markdown_blocks(response)
        with open(f'{working_dir_path}\\{file_path}', 'w') as f:
            f.write(content.replace(old_code, new_code))

    def delete(file_path):
        os.remove(f"{working_dir_path}\\{file_path}")

    for action, file, prompt in actions:
        if action == 'WRITE':
            repeat_until_finish(lambda: write(file, prompt), max_retries=2)
        elif action == 'DELETE':
            delete(file)
        elif action == 'EDIT':
            repeat_until_finish(lambda: edit(file, prompt), max_retries=2)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
