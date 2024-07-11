import networkx as nx

def collapse_sccs(G: nx.DiGraph):
    # for each strongly connected component, collapse it into a single node whose value is all the strings of each node concatenated
    sccs = nx.strongly_connected_components(G)
    for scc in sccs:
        if len(scc) > 1:
            new_node = ''.join([G.nodes[node]['label'] for node in scc])
            G.add_node(new_node)
            G.remove_nodes_from(scc)
            for pred in G.predecessors(scc.pop()):
                G.add_edge(pred, new_node)
    return G

def all_topological_sorts(G: nx.DiGraph):
    return list(nx.all_topological_sorts(G))