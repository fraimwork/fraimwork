import networkx as nx
import os
from functools import lru_cache

class FileTree(nx.DiGraph):
    def reverse_level_order(self):
        return list(nx.topological_sort(self))[::-1]
    
    def root_node(self):
        return '.'
    
    @lru_cache
    def leaf_nodes(self):
        return [node for node in self.nodes if not list(self.successors(node))]
    
    @staticmethod
    def from_directory(root_path):
        return build_file_tree_dag(root_path)
    
    @lru_cache
    def __str__(self):
        return filetree_to_string(self)
    
    def __repr__(self):
        return self.__str__()

def build_file_tree_dag(root_path):
    # Initialize a directed graph
    dag = FileTree()
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            parent_dir = os.path.relpath(dirpath, root_path)
            child_dir = os.path.relpath(os.path.join(dirpath, dirname), root_path)
            for node in [parent_dir, child_dir]:
                dag.add_node(node, name=os.path.basename(node), path=f'{root_path}/{node}')
            dag.add_edge(parent_dir, child_dir)
        for filename in filenames:
            parent_dir = os.path.relpath(dirpath, root_path)
            child_file = os.path.relpath(os.path.join(dirpath, filename), root_path)
            for node in [parent_dir, child_file]:
                dag.add_node(node, name=os.path.basename(node), path=f'{root_path}/{node}')
            with open(f'{root_path}/{child_file}', 'r') as f:
                content = f.read()
                dag.nodes[child_file]['content'] = content
            dag.add_edge(parent_dir, child_file)
    return dag

def filetree_to_string(tree: FileTree):
    def dfs(node, indent=''):
        result = ''
        items = tree[node]
        for item in items:
            item_attr = tree.nodes[item]
            item_path = item_attr['path']
            if os.path.isdir(item_path):
                result += f"{indent}├── {item_attr['name']}\\\n"
                result += dfs(item, indent + '│   ')
            else:
                result += f"{indent}├── {item_attr['name']}\n"
        return result
    return dfs(tree.root_node())

def write_file_tree(file_tree_string, base_dir):
    lines = file_tree_string.split('\n')
    stack = [base_dir]
    
    for line in lines:
        indent_level = len(line) - len(line.lstrip(' │'))
        current_dir = stack[indent_level // 4]  # Assuming 4 spaces per indentation level
        new_path = os.path.join(current_dir, line.split(' ')[-1])
        
        if line.endswith('\\'):
            os.makedirs(new_path, exist_ok=True)
            if len(stack) > indent_level // 4 + 1:
                stack[indent_level // 4 + 1] = new_path
            else:
                stack.append(new_path)
        else:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            open(new_path, 'w').close()