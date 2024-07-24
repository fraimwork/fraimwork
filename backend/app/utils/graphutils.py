import networkx as nx

def kosarajus(G: nx.DiGraph) -> nx.DiGraph:
    """
    Given a directed graph G, uses Kosaraju's algorithm to find strongly connected components
    and constructs a directed acyclic graph (DAG) by collapsing these components into single nodes.
    The attributes of each node in the DAG are set to be the subnodes (nodes in the original graph) 
    that belong to that SCC.

    Args:
        G: The input directed graph.

    Returns:
        A directed acyclic graph (DAG) derived from the input graph, with subnodes as attributes.
    """

    # Find strongly connected components using Kosaraju's algorithm
    scc = list(nx.strongly_connected_components(G))

    # Create a new DAG
    dag = nx.DiGraph()

    # Add nodes to the DAG (one for each SCC) and set subnodes as attributes
    for i, component in enumerate(scc):
        dag.add_node(i, subnodes=list(component))  # Set subnodes as an attribute

    # Add edges to the DAG based on connections between SCCs
    for u, v in G.edges():
        u_scc = next((i for i, comp in enumerate(scc) if u in comp), None)
        v_scc = next((i for i, comp in enumerate(scc) if v in comp), None)
        if u_scc != v_scc:
            dag.add_edge(u_scc, v_scc)

    return dag

def dag_to_levels(dag: nx.DiGraph):
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Input graph is not a DAG")
    level_arr = []
    graph_copy = dag.copy()
    sources = set([node for node in graph_copy.nodes if graph_copy.in_degree(node) == 0])
    while len(graph_copy.nodes) > 0:
        level_arr.append(sources)
        neighbors = [neighbor for source in sources for neighbor in graph_copy[source]]
        graph_copy.remove_nodes_from(sources)
        sources = set([node for node in neighbors if graph_copy.in_degree(node) == 0])
    return level_arr

def loose_level_order(G: nx.DiGraph):
    '''
    Given a directed graph G, returns a list of levels where each level is a list of sets denoting SCCs.
    '''
    SCC = kosarajus(G)
    levels = dag_to_levels(SCC)
    return [[set(SCC.nodes[node]['subnodes']) for node in level] for level in levels]


def remove_cycle_from_digraph(G):
    try:
        # Find a cycle in the graph
        cycle = nx.find_cycle(G, orientation='original')
        
        # Find an edge to remove based on node sizes
        for u, v, _ in cycle:
            if len(G.nodes[u]['content']) > len(G.nodes[v]['size']):
                G.remove_edge(u, v)
                return True  # Cycle was found and an edge was removed
    except nx.NetworkXNoCycle:
        pass  # No cycle found
    
    return False  # No cycle was found

def make_dag(G):
    while remove_cycle_from_digraph(G):
        pass
