from abc import ABC, abstractmethod
import random
from collections import defaultdict
from queue import PriorityQueue
import networkx as nx

class Cache(ABC):
    def __init__(self, capacity) -> None:
        self.capacity = capacity
    # x in cache
    @abstractmethod
    def __contains__(self, x):
        pass

    # cache[x]
    @abstractmethod
    def __getitem__(self, x):
        pass

    @abstractmethod
    def query(self, x):
        pass

class PageRankCache(Cache):
    def __init__(self, capacity, graph) -> None:
        super().__init__(capacity)
        self.page_rank = nx.pagerank(graph)
        # cache the top capacity nodes
        self.cache = set(sorted(self.page_rank, key=self.page_rank.get, reverse=True)[:capacity])

    def __contains__(self, x):
        return x in self.cache

    def __setitem__(self, x, y):
        pass

    def __getitem__(self, x):
        return x in self.cache

class LFUCache(Cache):
    def __init__(self, capacity) -> None:
        super().__init__(capacity)
        self.cache = {}
        self.freq = defaultdict(int)
        self.time = 0
        self.pq = PriorityQueue()
        self.min_freq = 0

    def __contains__(self, x):
        return x in self.cache

    def __setitem__(self, x, y):
        if self.capacity == 0:
            return
        if x in self.cache:
            self.cache[x] = y
            self.freq[x] += 1
            self.pq.put((self.time, self.min_freq, x))
            self.time += 1
        else:
            if len(self.cache) == self.capacity:
                while self.pq:
                    _, freq, node = self.pq.get()
                    if self.freq[node] == freq:
                        del self.cache[node]
                        del self.freq[node]
                        break
            self.cache[x] = y
            self.freq[x] = 1
            self.pq.put((self.time, 1, x))
            self.time += 1
            self.min_freq = 1

    def __getitem__(self, x):
        if x in self.cache:
            self.freq[x] += 1
            self.pq.put((self.time, self.min_freq, x))
            self.time += 1
            return True
        return False

class LRUCache(Cache):
    def __init__(self, capacity) -> None:
        super().__init__(capacity)
        self.cache = {}
        self.time = 0
        self.pq = PriorityQueue()

    def __contains__(self, x):
        return x in self.cache

    def __setitem__(self, x, y):
        if self.capacity == 0:
            return
        if x in self.cache:
            self.cache[x] = y
            self.pq.put((self.time, x))
            self.time += 1
        else:
            if len(self.cache) == self.capacity:
                _, node = self.pq.get()
                del self.cache[node]
            self.cache[x] = y
            self.pq.put((self.time, x))
            self.time += 1

    def __getitem__(self, x):
        if x in self.cache:
            self.pq.put((self.time, x))
            self.time += 1
            return True
        return False