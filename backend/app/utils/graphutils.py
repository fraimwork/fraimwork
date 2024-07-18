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

