from utils.filetreeutils import FileTree
from abc import ABC, abstractmethod
import networkx as nx
import os, re

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
        import_pattern = re.compile(r"import\s+'[^:]*:(?:[^/]+/)?([^']+)';")
        # Read all the .dart files in the directory
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                file_path = f'{dirpath}/{filename}'
                relative_path = os.path.relpath(file_path, self.path)
                u = os.path.relpath(os.path.join(dirpath, filename), self.path)
                if any(filename.endswith(ext) for ext in ['.dart']):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        for imp in import_pattern.findall(content):
                            # First try the import as an absolute path
                            if os.path.exists(absimppath := os.path.join(self.path, imp)):
                                G.add_edge(u, os.path.relpath(absimppath, self.path))
                            # Next try the import as a relative path
                            elif os.path.exists(relimppath := os.path.join(dirpath, imp)):
                                G.add_edge(u, os.path.relpath(relimppath, self.path))
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
