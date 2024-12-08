from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict, OrderedDict
class Cache(ABC):
    def __init__(self, capacity):
        self.capacity = capacity
        self.graph = None
        self.hits = 0
        self.misses = 0
    
    @abstractmethod
    def initialize(self, graph):
        """Initialize the cache with a graph."""
        self.graph = graph

    @abstractmethod
    def query(self, node):
        """Query the cache with a node."""
        pass

    def hit_rate(self):
        """Calculate and return the hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

from collections import OrderedDict

class LRUCache(Cache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = OrderedDict()

    def initialize(self, graph):
        self.cache.clear()
        super().initialize(graph)
        page_rank = nx.pagerank(self.graph)
        # cache the top capacity nodes by PageRank
        for node, _ in sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:self.capacity]:
            self.cache[node] = None

    def query(self, node):
        if node in self.cache:
            # Cache hit
            self.hits += self.graph.nodes[node]['size']
            # Move the accessed node to the end to show it was recently used
            self.cache.move_to_end(node)
            return self.cache[node]
        else:
            # Cache miss
            self.misses += self.graph.nodes[node]['size']
            if len(self.cache) >= self.capacity:
                # Remove the least recently used item
                self.cache.popitem(last=False)
            # Add the new node to the cache
            self.cache[node] = None
            return self.cache[node]

class LFUCache(Cache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = {}
        self.freq = defaultdict(int)
    
    def initialize(self, graph):
        self.cache.clear()
        self.freq.clear()
        super().initialize(graph)
        page_rank = nx.pagerank(self.graph)
        # cache the top capacity nodes by PageRank
        for node, _ in sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:self.capacity]:
            self.cache[node] = None
            self.freq[node] = 0

    def query(self, node):
        if node in self.cache:
            # Cache hit
            self.hits += self.graph.nodes[node]['size']
            # Increment the frequency of the node by 'size'
            self.freq[node] += 1
            return self.cache[node]
        else:
            # Cache miss
            self.misses += self.graph.nodes[node]['size']
            if len(self.cache) >= self.capacity:
                # Find the least frequently used node
                lfu_node = min(self.freq, key=self.freq.get)
                # Remove it from the cache and frequency dictionary
                del self.cache[lfu_node]
                del self.freq[lfu_node]
            # Add the new node to the cache and set its frequency to 1
            self.cache[node] = None
            self.freq[node] = 1
            return self.cache[node]
class WeightedLFUCache(Cache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = {}
        self.freq = defaultdict(int)
    
    def initialize(self, graph):
        self.cache.clear()
        self.freq.clear()
        super().initialize(graph)
        page_rank = nx.pagerank(self.graph)
        # cache the top capacity nodes by PageRank
        for node, _ in sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:self.capacity]:
            self.cache[node] = None
            self.freq[node] = 0

    def query(self, node):
        if node in self.cache:
            # Cache hit
            self.hits += self.graph.nodes[node]['size']
            # Increment the frequency of the node by 'size'
            self.freq[node] += self.graph.nodes[node]['size']
            return self.cache[node]
        else:
            # Cache miss
            self.misses += self.graph.nodes[node]['size']
            if len(self.cache) >= self.capacity:
                # Find the least frequently used node
                lfu_node = min(self.freq, key=self.freq.get)
                # Remove it from the cache and frequency dictionary
                del self.cache[lfu_node]
                del self.freq[lfu_node]
            # Add the new node to the cache and set its frequency to 'size'
            self.cache[node] = None
            self.freq[node] = self.graph.nodes[node]['size']
            return self.cache[node]

class PageRankCache(Cache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = {}
    
    def initialize(self, graph):
        self.cache.clear()
        super().initialize(graph)
        page_rank = nx.pagerank(self.graph)
        # cache the top capacity nodes by PageRank
        for node, _ in sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:self.capacity]:
            self.cache[node] = None

    def query(self, node):
        if node in self.cache:
            # Cache hit
            self.hits += self.graph.nodes[node]['size']
            return self.cache[node]
        else:
            # Cache miss
            self.misses += self.graph.nodes[node]['size']
            return None

class WeightedPageRankCache(Cache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = {}
    
    def initialize(self, graph):
        self.cache.clear()
        super().initialize(graph)
        personalization = {node: self.graph.nodes[node]['size'] for node in self.graph.nodes}
        page_rank = nx.pagerank(self.graph, personalization=personalization)
        # cache the top capacity nodes by PageRank
        for node, _ in sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:self.capacity]:
            self.cache[node] = None

    def query(self, node):
        if node in self.cache:
            # Cache hit
            self.hits += self.graph.nodes[node]['size']
            return self.cache[node]
        else:
            # Cache miss
            self.misses += self.graph.nodes[node]['size']
            return None

class InDegreeCache(Cache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = {}
    
    def initialize(self, graph):
        self.cache.clear()
        super().initialize(graph)
        # cache the top capacity nodes by in-degree
        for node, degree in sorted(self.graph.in_degree, key=lambda x: x[1], reverse=True)[:self.capacity]:
            self.cache[node] = None

    def query(self, node):
        if node in self.cache:
            # Cache hit
            self.hits += self.graph.nodes[node]['size']
            return self.cache[node]
        else:
            # Cache miss
            self.misses += self.graph.nodes[node]['size']
            return None

class WeightedInDegreeCache(Cache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = {}
    
    def initialize(self, graph):
        self.cache.clear()
        super().initialize(graph)
        inDegrees = dict(self.graph.in_degree)
        weightedInDegrees = {node: inDegrees[node] * self.graph.nodes[node]['size'] for node in self.graph.nodes}
        # cache the top capacity nodes by weighted in-degree
        for node, degree in sorted(weightedInDegrees.items(), key=lambda x: x[1], reverse=True)[:self.capacity]:
            self.cache[node] = None

    def query(self, node):
        if node in self.cache:
            # Cache hit
            self.hits += self.graph.nodes[node]['size']
            return self.cache[node]
        else:
            # Cache miss
            self.misses += self.graph.nodes[node]['size']
            return None

class TopKSizeCache(Cache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = {}
    
    def initialize(self, graph):
        self.cache.clear()
        super().initialize(graph)
        # cache the top capacity nodes by size
        for node, size in sorted(nx.get_node_attributes(self.graph, 'size').items(), key=lambda x: x[1], reverse=True)[:self.capacity]:
            self.cache[node] = None

    def query(self, node):
        if node in self.cache:
            # Cache hit
            self.hits += self.graph.nodes[node]['size']
            return self.cache[node]
        else:
            # Cache miss
            self.misses += self.graph.nodes[node]['size']
            return None