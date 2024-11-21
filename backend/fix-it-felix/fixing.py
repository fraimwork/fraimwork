import google.generativeai as genai
from dependency import build_dependency_graph
from dependency import topological_sort
import difflib
from logic.utils.stringutils import compute_nested_levels, linewise_tokenize, wordwise_tokenize

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
    context_prompt = f"""You are a software engineer tasked with editing a large code file to implement changes in accordance with the instructions you will receive in the following prompts. Large edits at once are NOT the way. We will solve this problem by dividing the edit into smaller sub-edits consisting of additions and deletions (no more than 10 lines per edit). We will use a custom file type called 'skinnydiff', which are diff-like files but only feature a larger file's substring, to convey these changes. If there are no changes you'd like to make for step N, then you don't have to output a skinnydiff. Here is an example of a 'skinnydiff' file:
    skinnydiff
        void foo() {{
        - int x = 5;
        + var x = 7;
        return x + 3; // Example code 
        }}

        Skinnydiffs are shorter versions of diff files that only include a few context lines around the deletions and insertions.

        Given this knowledge, structure your response as follows:

        # Step 1
        $Talk through the 1st sub-change and what you plan to do$
        ## Scope
        $Brief planning on which line or class to include in the change$
        ## Change 1
        
    skinnydiff
        $code diff$


        # Step N
        $Talk through the Nth sub-change and what you plan to do$
        ## Scope
        $Brief planning on which line or class to include in the change$
        ## Change N
        
    skinnydiff
        $...$
    """

    fix_code_prompt = f"Here is a broken code:\n\n{broken_code}\n\n" \
             f"Here is the context from its dependencies:\n\n{context_str}\n\n" \
             f"Fix the broken code considering the context, and try to perserve the style of the current code. \n\n" \
             f"Address any bugs or errors present in the code.\n\n" \
             f"Do NOT offer explanations, just the correct code. \n\n" \
             f"Do NOT use ANY formatting character. \n\n" \
             f"Do not modify or remove any comments in the file. \n\n" \
    
    # model: gemini-1.5-flash
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model.generate_content(context_prompt + fix_code_prompt) 
    
    if response and response.text:
        diffs_text = response.text.strip()
        # Parse the response into individual diff sections
        diffs = parse_skinnydiffs(diffs_text)
        # output = join_diff(broken_code.splitlines(), diffs)
        print("HERE IS THE OUTPUT")
        print("\n", diffs)
        return "haha"
        # return output
    else:
        return "No response received."


def parse_skinnydiffs(diffs_text):
    """
    Parses the text returned by the API into structured diff sections.
    """
    inserts = {"scope": "", "lines": []}

    tokens = linewise_tokenize(diffs_text)
    print(tokens)

        
    return diffs

def join_diff(code, diffs):
    #todo: find correct places to insert diff
    return diffs

# Helper function to read file content (mocked for now)
def read_file(file):
    # In a real-world scenario, read from disk or a database
    with open(file, 'r') as f:
        return f.read()

# Helper function to write file content (used to directly edit broken file with the code gemini returns)
def write_file(file, content):
    with open(file, 'w') as f:
        f.write(content)
