import google.generativeai as genai
from dependency import build_dependency_graph
from dependency import topological_sort

# Fixes broken code based on its dependencies
def fix_broken_code(dependency_graph, broken_nodes):
    fixed_code = {}
    
    # Build the dependency graph using NetworkX
    G = build_dependency_graph(dependency_graph)

    # Topologically sort the graph to fix dependencies first
    sorted_nodes = topological_sort(G)
    
    # Filter the sorted nodes to only include broken nodes
    nodes_to_fix = [node for node in sorted_nodes if node in broken_nodes]
    
    # Iterate over each broken node
    for node in nodes_to_fix:
        # Gather dependencies (predecessors) of the broken node
        dependencies = list(G.predecessors(node))
        
        # Read the broken file and dependency contents (you'll need to define read_file)
        broken_file_content = read_file(node)
        context = {dep: read_file(dep) for dep in dependencies}

        # Call LLM to fix the broken file considering its dependencies
        fixed_file_content = call_llm_to_fix(broken_file_content, context)
        
        # Store the fixed code
        fixed_code[node] = fixed_file_content
        write_file(node, fixed_file_content)

    return fixed_code

# LLM API integration to fix code
def call_llm_to_fix(broken_code, context):
    API_KEY = "AIzaSyCLz8hJEC1iSfVb0KWDYink4C61f8hpXQU" # Make into env variable eventually...
    genai.configure(api_key=API_KEY)

    # Create a prompt combining broken code and its context
    context_str = "\n\n".join([f"Dependency {dep}: {content}" for dep, content in context.items()])
    prompt = f"Here is a broken code:\n\n{broken_code}\n\n" \
             f"Here is the context from its dependencies:\n\n{context_str}\n\n" \
             f"Fix the broken code considering the context, and try to perserve the style of the current code. \n\n" \
             f"Address any bugs or errors present in the code.\n\n" \
             f"Do NOT offer explanations, just the correct code. \n\n" \
             f"Do NOT use ANY formatting character. \n\n" \
             f"Do not modify or remove any comments in the file. \n\n" \
            #  f"Do NOT any extra explanation. \n\n"
    
    # model: gemini-1.5-flash
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt) # maybe can add temperature and token settings?
    print(response.text) # just to see what it responds with... remove in later iterations

    return response.text.strip()

# Helper function to read file content (mocked for now)
def read_file(file):
    # In a real-world scenario, read from disk or a database
    with open(file, 'r') as f:
        return f.read()

# Helper function to write file content (used to directly edit broken file with the code gemini returns)
def write_file(file, content):
    with open(file, 'w') as f:
        f.write(content)
