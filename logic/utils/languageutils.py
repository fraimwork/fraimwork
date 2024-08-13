from abc import ABC, abstractmethod
import os, re, networkx as nx
from types import NoneType

def patterns(ext):
    match ext:
        case '.py', '.ipynb':
            return re.compile(r'^\s*(import\s+[^#\n]+|from\s+[^#\n]+\s+import\s+[^#\n]+)', re.VERBOSE | re.MULTILINE)
        case '.swift':
            return re.compile(r'^\s*import\s+[\w.]+', re.MULTILINE)
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
        case '.json':
            # Match the whole thing
            return re.compile(r'^\s*"(?:[^"\\]|\\.)*"\s*:\s*"(?:[^"\\]|\\.)*"', re.MULTILINE)
        case _: return None

def get_imports(file_path: str) -> list[str] | NoneType:
    _, ext = os.path.splitext(file_path)
    pattern = patterns(ext)
    if not pattern: return None
    imports = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        imports.extend(patterns(ext).findall(content))
    # Limit imports length to 50
    imports = imports[:50]
    return list(set(imports))
