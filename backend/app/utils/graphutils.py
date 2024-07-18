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
    level_dict = []
    graph_copy = dag.copy()
    sources = set([node for node in graph_copy.nodes if graph_copy.in_degree(node) == 0])
    while len(graph_copy.nodes) > 0:
        level_dict.append(sources)
        neighbors = [neighbor for source in sources for neighbor in graph_copy[source]]
        graph_copy.remove_nodes_from(sources)
        sources = set([node for node in neighbors if graph_copy.in_degree(node) == 0])
    return level_dict

def loose_level_order(G: nx.DiGraph):
    G = kosarajus(G)
    levels = dag_to_levels(G)
    return [[set([subnode for subnode in G.nodes[node]['subnodes']]) for node in level] for level in levels]

