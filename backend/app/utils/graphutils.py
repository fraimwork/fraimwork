import networkx as nx
import os
from utils.stringutils import extract_filename

def build_file_tree_dag(root_path):
    # Initialize a directed graph
    dag = nx.DiGraph()
    
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            parent_dir = os.path.relpath(dirpath, root_path)
            child_dir = os.path.relpath(os.path.join(dirpath, dirname), root_path)
            dag.add_node(parent_dir, name=extract_filename(parent_dir), path=parent_dir)
            dag.add_node(child_dir, name=extract_filename(child_dir), path=child_dir)
            dag.add_edge(parent_dir, child_dir)
        for filename in filenames:
            parent_dir = os.path.relpath(dirpath, root_path)
            child_file = os.path.relpath(os.path.join(dirpath, filename), root_path)
            dag.add_node(parent_dir, name=extract_filename(parent_dir), path=parent_dir)
            with open(f'{root_path}/{child_file}', 'r') as f:
                content = f.read()
            dag.add_node(child_file, name=extract_filename(child_file), path=child_file, content=content)
            dag.add_edge(parent_dir, child_file)
    
    return dag

def string_represented_file_tree(graph: nx.DiGraph):
    # Initialize a string to store the tree representation
    tree = ""
    
    # Walk through the graph
    for node in graph.nodes:
        # Get the parent node
        parent = list(graph.predecessors(node))
        
        # If the node is a directory
        if node.endswith('/'):
            # Add the directory to the tree representation
            tree += f"{node}\n"
        # If the node is a file
        else:
            # Add the file to the tree representation
            tree += f"├── {node}\n"
    
    return tree