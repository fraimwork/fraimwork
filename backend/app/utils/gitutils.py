import os, json, requests, git
from utils.filetreeutils import FileTree

def clone_repo(repo_url: str):
    pieces = repo_url.replace('.git', '').split('/')
    repo_name = pieces[-1]
    user_name = pieces[-2]
    local_repo_path = f'./tmp/{user_name}/{repo_name}'
    return git.Repo.clone_from(repo_url, local_repo_path)

def create_branch(repo, branch_name):
    repo.git.checkout('main')
    repo.git.checkout('-b', branch_name)

def make_directories_from_tree(repo, tree: FileTree):
    for node in tree.nodes:
        if not os.path.exists(node['path']):
            os.makedirs(node)
    for edge in tree.edges:
        parent, child = edge
        with open(parent, 'w') as f:
            f.write(f'// {parent} contents')
        repo.git.add(parent)
        repo.index.commit(f'Add {parent}')
    return repo

def create_pull_request(repo_link, base_branch, new_branch, title, body):
    return 'https://google.com'
    # Obtain our GitHub personal access token
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

    # Construct the PR
    BASE_BRANCH = base_branch
    NEW_BRANCH = new_branch
    PR_TITLE = title
    PR_BODY = body

    # Set the headers for authorization
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Step 1: Get the SHA of the base branch (main)
    response = requests.get(
        f'{repo_link}/branches/{BASE_BRANCH}',
        headers=headers
    )
    response.raise_for_status()
    base_sha = response.json()['commit']['sha']

    # Step 2: Create a new branch
    data = {
        'ref': f'refs/heads/{NEW_BRANCH}',
        'sha': base_sha
    }
    response = requests.post(
        f'https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/git/refs',
        headers=headers,
        data=json.dumps(data)
    )
    response.raise_for_status()

    # Note: You would typically make some changes to the new branch here before committing and pushing

    # Step 3: Create a pull request
    data = {
        'title': PR_TITLE,
        'body': PR_BODY,
        'head': NEW_BRANCH,
        'base': BASE_BRANCH
    }
    response = requests.post(
        f'https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/pulls',
        headers=headers,
        data=json.dumps(data)
    )
    response.raise_for_status()
    pr_url = response.json()['html_url']

    print(f'Pull request created: {pr_url}')