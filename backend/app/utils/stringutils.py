import re
from collections import defaultdict

def arr_from_sep_string(string: str, sep=','):
    return [x.strip() for x in string.split(sep)]

def arr_from_numbered_list(string: str):
    withNumbers = arr_from_sep_string(string, "\n")
    return [x.split(' ')[1] for x in withNumbers]

def extract_markdown_blocks(text):
    """
    Extract markdown blocks from a string.

    Parameters:
    text (str): The input string containing markdown blocks.

    Returns:
    list: A list of strings, each representing a markdown block.
    """
    pattern = re.compile(r"```(.*?)\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(text)
    blocks = [block[1].strip() for block in blocks]
    return blocks

def extract_snippet(text, snippet, padding):
    """
    Extract a snippet from a string.

    Parameters:
    text (str): The input string containing the snippet.
    snippet (str): The snippet to extract.
    padding (int): The number of lines to include before and after the snippet.

    Returns:
    str: The extracted snippet.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if snippet in line:
            start = max(0, i - padding)
            end = min(len(lines), i + padding + 1)
            return '\n'.join(lines[start:end])
    return None

def markdown_to_dict(markdown: str) -> dict:
    # Regular expression to match headers
    header_regex = re.compile(r'^(#+)\s*(.*)', re.MULTILINE)
    
    # Dictionary to store the result
    result = defaultdict(str)
    
    # Find all headers
    headers = [(m.group(1).count('#'), m.group(2), m.start()) for m in header_regex.finditer(markdown)]
    
    # Sort headers by their position in the text
    headers.sort(key=lambda x: x[2])
    
    for i in range(len(headers)):
        current_header = headers[i]
        header_level, header_text, header_start = current_header
        
        # Find the end of the current header's content
        if i + 1 < len(headers):
            next_header_start = headers[i + 1][2]
            content = markdown[header_start + len(current_header[1]) + header_level + 1:next_header_start].strip()
        else:
            content = markdown[header_start + len(current_header[1]) + header_level + 1:].strip()
        
        # Add to result dictionary
        result[header_text.lower()] = content
    
    return dict(result)