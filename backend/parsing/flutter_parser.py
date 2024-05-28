import networkx as nx
import re

def parse_flutter_code(code):
    graph = nx.DiGraph()
    # split code into lines by newline character and semi-colon
    lines = re.split('\n|;', code)
    return lines