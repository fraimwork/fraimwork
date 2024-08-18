const { createGraph, alg } = require('graphlib');  // Graphlib for graph functionality
const _ = require('lodash');

function kosarajus(G) {
    /**
     * Given a directed graph G, uses Kosaraju's algorithm to find strongly connected components
     * and constructs a directed acyclic graph (DAG) by collapsing these components into single nodes.
     * The attributes of each node in the DAG are set to be the subnodes (nodes in the original graph) 
     * that belong to that SCC.
     */
    
    // Find strongly connected components using Kosaraju's algorithm
    const scc = alg.kosarajuSCC(G);

    // Create a new DAG
    const dag = createGraph();

    // Add nodes to the DAG (one for each SCC) and set subnodes as attributes
    scc.forEach((component, i) => {
        dag.setNode(i, { subnodes: component });
    });

    // Add edges to the DAG based on connections between SCCs
    G.edges().forEach(({ v: u, w: v }) => {
        const u_scc = scc.findIndex(comp => comp.includes(u));
        const v_scc = scc.findIndex(comp => comp.includes(v));
        if (u_scc !== v_scc) {
            dag.setEdge(u_scc, v_scc);
        }
    });

    return dag;
}

function dagToLevels(dag) {
    if (!alg.isAcyclic(dag)) {
        throw new Error("Input graph is not a DAG");
    }
    
    const levelArr = [];
    const graphCopy = dag.clone();
    let sources = graphCopy.nodes().filter(node => graphCopy.inEdges(node).length === 0);
    
    while (graphCopy.nodeCount() > 0) {
        levelArr.push(sources);
        const neighbors = _.flatten(sources.map(source => graphCopy.successors(source)));
        graphCopy.removeNodes(sources);
        sources = _.uniq(neighbors.filter(node => graphCopy.inEdges(node).length === 0));
    }

    return levelArr;
}

function collapsedLevelOrder(G) {
    /**
     * Given a directed graph G, returns a list of levels where each level is a list of sets denoting SCCs.
     */
    const SCC = kosarajus(G);
    const levels = dagToLevels(SCC);
    return levels.map(level => level.map(node => new Set(SCC.node(node).subnodes)));
}

function looseLevelOrder(G, key = "content") {
    /**
     * Given a directed graph G, returns a list of levels where each level is a list of nodes in that level.
     */
    const dag = G.clone();
    makeDAG(dag, key);
    return dagToLevels(dag);
}

function removeCycleFromDiGraph(G, key) {
    try {
        // Find a cycle in the graph
        const cycle = alg.findCycles(G)[0];  // Assuming there's at least one cycle

        // Find an edge to remove based on node sizes
        for (let i = 0; i < cycle.length - 1; i++) {
            const u = cycle[i];
            const v = cycle[i + 1];
            if (G.node(u)[key].length > G.node(v)[key].length) {
                G.removeEdge(u, v);
                return true;  // Cycle was found and an edge was removed
            }
        }
    } catch (error) {
        // No cycle found
    }
    
    return false;  // No cycle was found
}

function makeDAG(G, key) {
    while (removeCycleFromDiGraph(G, key)) {
        // Continue until all cycles are removed
    }
}

function mstFromNode(G, node) {
    /**
     * Given a graph G and a node, returns the minimum spanning tree of the graph with the given node as the root.
     * Using Prim's algorithm for MST.
     */
    // Unfortunately, graphlib doesn't have built-in MST algorithms.
    // You would need to implement Prim's algorithm manually or use another library for this purpose.
    // Here's a simplified version (not optimal for large graphs) to mimic the MST calculation:
    
    const mst = createGraph({ directed: false });
    const visited = new Set([node]);
    const edges = [];

    G.edges().forEach(edge => {
        if (edge.v === node || edge.w === node) {
            edges.push(edge);
        }
    });

    while (visited.size < G.nodeCount()) {
        edges.sort((a, b) => G.edge(a) - G.edge(b));  // Sort by edge weight
        const edge = edges.shift();
        
        if (!visited.has(edge.w)) {
            mst.setEdge(edge.v, edge.w, G.edge(edge));
            visited.add(edge.w);

            G.successors(edge.w).forEach(neighbor => {
                if (!visited.has(neighbor)) {
                    edges.push({ v: edge.w, w: neighbor, weight: G.edge(edge.w, neighbor) });
                }
            });
        }
    }

    return mst;
}

module.exports = {
    kosarajus,
    dagToLevels,
    collapsedLevelOrder,
    looseLevelOrder,
    removeCycleFromDiGraph,
    makeDAG,
    mstFromNode
};
