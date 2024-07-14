import os, json, requests, git
from utils.filetreeutils import FileTree

def get_repo_size(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url)
    if response.status_code == 200:
        repo_info = response.json()
        size_kb = repo_info.get('size', 0)
        size_mb = size_kb / 1024
        return size_mb
    else:
        print(f"Failed to fetch repository info: {response.status_code}")
        return None

def clone_repo(repo_url: str, max_size_mb=100):
    pieces = repo_url.replace('.git', '').split('/')
    repo_name = pieces[-1]
    user_name = pieces[-2]
    size_mb = get_repo_size(user_name, repo_name)
    if size_mb is None or size_mb > max_size_mb:
        print(f"Repository is too large: {size_mb} MB")
        return None
    local_repo_path = f'./tmp/{user_name}/{repo_name}'
    if os.path.exists(local_repo_path):
        return git.Repo(local_repo_path)
    return git.Repo.clone_from(repo_url, local_repo_path)

def create_branch(repo, base_name, branch_name):
    repo.git.checkout(base_name)
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

def create_pull_request(repo, base_branch, new_branch, title, body, token):
    """
    Creates a pull request on GitHub.

    :param repo: git.Repo object
    :param base_branch: The branch you want to merge into (usually 'main' or 'master')
    :param new_branch: The branch you want to merge from
    :param title: The title of the pull request
    :param body: The body description of the pull request
    :param token: GitHub personal access token
    :return: Response from the GitHub API
    """
    # Extract the repository owner and name from the remote URL
    remote_url = repo.remotes.origin.url
    if remote_url.startswith('git@'):
        remote_url = remote_url.replace(':', '/').replace('git@', 'https://')
    elif remote_url.startswith('https://'):
        remote_url = remote_url.replace('.git', '')
    
    parts = remote_url.split('/')
    owner = parts[-2]
    repo_name = parts[-1]

    # GitHub API URL for creating a pull request
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls"

    # Headers for authentication
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Payload for the pull request
    data = {
        "title": title,
        "body": body,
        "head": new_branch,
        "base": base_branch
    }

    # Make the request to create the pull request
    response = requests.post(api_url, headers=headers, data=json.dumps(data))

    if response.status_code == 201:
        print("Pull request created successfully!")
    else:
        print(f"Failed to create pull request: {response.status_code}")
        print(response.json())

    return response.json()