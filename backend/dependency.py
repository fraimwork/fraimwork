import networkx as nx

# Converts the given dependency graph to a NetworkX directed graph
def build_dependency_graph(dependency_dict):
    G = nx.DiGraph()  # Directed Graph (DAG)
    
    for node, dependencies in dependency_dict.items():
        for dep in dependencies:
            G.add_edge(dep, node)  # Dependency direction: dep -> node
    
    return G

# Function to perform topological sort using NetworkX
def topological_sort(graph):
    try:
        return list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("The dependency graph contains cycles, which cannot be processed.")