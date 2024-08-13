from abc import ABC, abstractmethod
import os, re, networkx as nx
from enum import Enum, auto
from types import NoneType

class FrameworkAnalyzer(ABC):
    def __init__(self, path, *extensions):
        self.path = path
        self.extensions = extensions
    
    @abstractmethod
    def buildDependencyGraph(self) -> nx.DiGraph:
        pass

class DartAnalyzer(FrameworkAnalyzer):
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
                with open(file_path, 'r') as f:
                    content = f.read()
                    G.add_node(u, content=content)
                if any(filename.endswith(ext) for ext in self.extensions):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        for imp in import_pattern.findall(content):
                            # First try the import as an absolute path
                            if os.path.exists(absimppath := os.path.join(self.path, imp)):
                                path = os.path.relpath(absimppath, self.path)
                                with open(absimppath, 'r') as f:
                                    content = f.read()
                                    G.add_node(path, content=content)
                                G.add_edge(u, path)
                            # Next try the import as a relative path
                            elif os.path.exists(relimppath := os.path.join(dirpath, imp)):
                                path = os.path.relpath(relimppath, self.path)
                                with open(relimppath, 'r') as f:
                                    content = f.read()
                                    G.add_node(path, content=content)
                                G.add_edge(u, path)
        return G


class NodeAnalyzer(FrameworkAnalyzer):
    def __init__(self, path):
        super().__init__(path, ['.js', '.jsx', '.ts', '.tsx'])
    
    def buildDependencyGraph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        import_pattern = re.compile(r"(?:import|require)\s*(?:\(\s*['\"]([^'\"]+)['\"]\s*\)|['\"]([^'\"]+)['\"])")
        
        # Read all the relevant files in the directory
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in self.extensions):
                    file_path = os.path.join(dirpath, filename)
                    u = os.path.relpath(file_path, self.path)
                    with open(file_path, 'r') as f:
                        content = f.read()
                        for match in import_pattern.findall(content):
                            imp = match[0] or match[1]  # One of the groups will be non-empty
                            # First try the import as an absolute path
                            if os.path.exists(absimppath := os.path.join(self.path, imp)):
                                G.add_edge(u, os.path.relpath(absimppath, self.path))
                            # Next try the import as a relative path
                            elif os.path.exists(relimppath := os.path.join(dirpath, imp)):
                                G.add_edge(u, os.path.relpath(relimppath, self.path))
                            # Handle node_modules imports
                            else:
                                node_modules_path = os.path.join(self.path, 'node_modules', imp)
                                if os.path.exists(node_modules_path):
                                    G.add_edge(u, os.path.relpath(node_modules_path, self.path))
        return G

class Framework(Enum):
    FLUTTER = "FLUTTER"
    REACT = "REACT"
    REACT_NATIVE = "REACT_NATIVE"
    ANGULAR = "ANGULAR"
    FIREBASE = "FIREBASE"
    FLASK = "FLASK"

    def __repr__(self) -> str:
        return self.value
    
    def __str__(self) -> str:
        return self.value

    @staticmethod
    def get_frameworks():
        return list(Framework.__members__.keys())
    
    @staticmethod
    def get_framework(framework_name: str):
        return Framework[framework_name.upper()]
    
    def get_analyzer(self, path) -> FrameworkAnalyzer | NoneType:
        match self:
            case Framework.FLUTTER:
                return DartAnalyzer(path)
            case Framework.REACT | Framework.REACT_NATIVE | Framework.ANGULAR:
                return NodeAnalyzer(path)
            case _:
                return None
            
    def get_working_dir(self) -> str:
        match self:
            case Framework.FLUTTER:
                return 'lib'
            case Framework.REACT | Framework.REACT_NATIVE | Framework.ANGULAR:
                return 'src'
            case Framework.FIREBASE:
                return 'functions\\src'
            case Framework.FLASK:
                return '.'
            case _:
                return None
    
    def get_file_extensions(self) -> list[str]:
        match self:
            case Framework.FLUTTER:
                return ['.dart']
            case Framework.REACT | Framework.REACT_NATIVE | Framework.ANGULAR | Framework.FIREBASE:
                return ['.js', '.jsx', '.ts', '.tsx']
            case Framework.FLASK:
                return ['.py']
            case _:
                return []

