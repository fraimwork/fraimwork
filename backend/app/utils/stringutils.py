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

def build_file_tree_dag(root_path):
    # Initialize a directed graph
    dag = nx.DiGraph()
    
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            parent_dir = os.path.relpath(dirpath, root_path)
            child_dir = os.path.relpath(os.path.join(dirpath, dirname), root_path)
            dag.add_edge(parent_dir, child_dir)
        for filename in filenames:
            parent_dir = os.path.relpath(dirpath, root_path)
            child_file = os.path.relpath(os.path.join(dirpath, filename), root_path)
            dag.add_edge(parent_dir, child_file)
    
    return dag
