from utils.filetreeutils import FileTree
from abc import ABC, abstractmethod
import networkx as nx
import os

class LanguageAnalyzer(ABC):
    def __init__(self, path, *extensions):
        self.path = path
        self.extensions = extensions
    
    @abstractmethod
    def buildDependencyGraph(self) -> nx.DiGraph:
        pass


class DartAnalyzer(LanguageAnalyzer):
    def __init__(self, path):
        super().__init__(path, '.dart')
    
    def buildDependencyGraph(self) -> nx.DiGraph:
        command = ['dart', 'fix', self.path, '--apply']
        os.system(' '.join(command))
        G = nx.DiGraph()
        # Read all the .dart files in the directory
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in self.extensions):
                    with open(f'{dirpath}/{filename}', 'r') as f:
                        content = f.read()
                        lines = content.split('\n')
                        # search for import statements
                        for line in lines:
                            if line.startswith('import'):
                                parts = line.split(' ')
                                imported_file = parts[1].replace(';', '')
                                G.add_edge(filename, imported_file)
        return G


class JavascriptAnalyzer(LanguageAnalyzer):
    def __init__(self, path):
        super().__init__(path, '.js', '.jsx')
    
    def buildDependencyGraph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        # Read all the .js and .jsx files in the directory
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in self.extensions):
                    with open(f'{dirpath}/{filename}', 'r') as f:
                        content = f.read()
                        lines = content.split('\n')
                        # search for import statements
                        for line in lines:
                            if line.startswith('import'):
                                parts = line.split(' ')
                                imported_file = parts[1].replace(';', '')
                                G.add_edge(filename, imported_file)
        return G
