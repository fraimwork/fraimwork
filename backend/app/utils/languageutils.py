from abc import ABC, abstractmethod
import os, re, networkx as nx
from types import NoneType
import json

def patterns(ext):
    match ext:
        case '.py':
            return re.compile(r"""
    ^\s*                                  # Start of line, allowing leading whitespace
    (?:import|from)\s+                    # 'import' or 'from' keyword followed by whitespace
    ([\w\.]+)                             # Captures the module name or module path
    (?:\s+as\s+\w+)?                      # Optionally matches 'as alias'
    (?:\s*,\s*[\w\.]+(?:\s+as\s+\w+)?)*   # Optionally matches multiple imports separated by commas
    (?:\s+import\s+[\w\.\(\)\s,]+)?       # Optionally matches 'from module import submodule(s)'
""", re.VERBOSE | re.MULTILINE)
        case '.dart':
            return re.compile(r"""
    ^\s*                                  # Start of line, allowing leading whitespace
    import\s+                             # 'import' keyword followed by whitespace
    ['"]                                  # Opening quote for the import path
    ([^'"]+)                              # Captures the import path
    ['"]                                  # Closing quote for the import path
    (?:\s+as\s+\w+)?                      # Optionally matches 'as alias'
    (?:\s+(?:show|hide)\s+[\w\s,]+)?      # Optionally matches 'show' or 'hide' with identifiers
    \s*;                                  # Ending with a semicolon, allowing trailing whitespace
""", re.VERBOSE | re.MULTILINE)
        case '.jsx' | '.js' | '.ts' | '.tsx':
            return re.compile(r"""
            ^\s*                                # Start of line, allowing leading whitespace
            import\s+                           # 'import' keyword followed by whitespace
            (?:
                (?:[\w*{}\s,]+)\s+from\s+       # Named imports or wildcard import followed by 'from'
                |
                (?=[^'"]*['"])                  # Lookahead to ensure there's a module specifier
            )
            ['"]                                # Opening quote for module specifier
            ([^'"]+)                            # Module specifier
            ['"]                                # Closing quote for module specifier
        """, re.VERBOSE | re.MULTILINE)
        case _: return None

def get_imports(file_path: str) -> list[str] | NoneType:
    _, ext = os.path.splitext(file_path)
    pattern = patterns(ext)
    if not pattern: return None
    imports = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if ext == '.ipynb':
            return None
            notebook_content = json.load(f)
            for cell in notebook_content.get('cells', []):
                if cell.get('cell_type') == 'code':
                    cell_source = ''.join(cell.get('source', []))
                    imports.extend(pattern.findall(cell_source))
        else:
            imports.extend(patterns(ext).findall(content))
    return list(set(imports))

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


class NodeAnalyzer(LanguageAnalyzer):
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
