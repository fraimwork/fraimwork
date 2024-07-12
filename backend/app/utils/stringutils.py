import networkx as nx
import os

def arr_from_sep_string(string: str, sep=","):
    return [x.strip() for x in string.split(sep)]

def generate_tree_structure(path, indent=''):
    result = ''
    items = os.listdir(path)
    for index, item in enumerate(items):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            result += f"{indent}├── {item}/\n"
            result += generate_tree_structure(item_path, indent + '│   ')
        else:
            result += f"{indent}├── {item}\n"
    return result

def dag_from_filetree(filetree: str):
    G = nx.DiGraph()
    lines = filetree.split("\n")
    for line in lines:
        if not line:
            continue
        depth = len(line) - len(line.lstrip())
        node = line.strip().replace("/", "")
        if depth == 0:
            G.add_node(node)
        else:
            parent = line[:depth].strip().replace("/", "")
            G.add_edge(parent, node)
    return G
