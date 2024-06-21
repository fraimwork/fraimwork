from abc import ABC, abstractmethod
import ast, esprima, json, subprocess
import networkx as nx
from functools import cache
from utils.graphs.dag_utils import collapse_sccs, all_topological_sorts
from itertools import product

class CodeParser(ABC):
    @abstractmethod
    @cache
    def get_ast(self, code) -> list:
        pass
    
    @cache
    def get_dependency_graph(self, code):
        G = nx.DiGraph()
        variables = {}
        # Traverse AST to identify variable declarations and their dependencies
        for statement in self.get_ast(code):
            if statement.type == 'VariableDeclaration':
                for declaration in statement.declarations:
                    var_name = declaration.id.name
                    var_line = declaration.loc.start.line
                    if declaration.init:
                        dependencies = self.find_dependencies(declaration.init)
                        G.add_node(var_line, label=var_name)
                        for dep in dependencies:
                            if dep not in variables: continue
                            G.add_edge(variables[dep], var_line)
                    else:
                        G.add_node(var_line, label=var_name)
                    variables[var_name] = var_line
        G = collapse_sccs(G)
        return G

    def find_dependencies(self, expression):
        dependencies = []
        if expression.type == 'BinaryExpression':
            dependencies += self.find_dependencies(expression.left)
            dependencies += self.find_dependencies(expression.right)
        elif expression.type == 'Identifier':
            dependencies.append(expression.name)
        elif expression.type == 'CallExpression':
            if expression.callee.type == 'Identifier':
                dependencies.append(expression.callee.name)
            for arg in expression.arguments:
                dependencies += self.find_dependencies(arg)
        return dependencies
    
    @cache
    def all_permutations(self, code):
        return all_topological_sorts(self.get_dependency_graph(code))

class PythonParser(CodeParser):
    def get_ast(self, code):
        return ast.parse(code).body

class JavascriptParser(CodeParser):
    def get_ast(self, code):
        return esprima.parseScript(code).body

class DartParser(CodeParser):
    def get_ast(self, code):
        process = subprocess.Popen(
            ['dart', 'generate_ast.dart'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=code)
        if process.returncode != 0:
            print(f"Error: {stderr}")
            return None
        return json.loads(stdout)