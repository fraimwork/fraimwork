def arr_from_sep_string(string: str, sep=","):
    return [x.strip() for x in string.split(sep)]

def extract_filename(path: str):
    return path.split("/")[-1]

import re

def extract_markdown_blocks(text):
    """
    Extract markdown blocks from a string.

    Parameters:
    text (str): The input string containing markdown blocks.

    Returns:
    list: A list of strings, each representing a markdown block.
    """
    # Regex to match markdown code blocks delimited by triple backticks
    pattern = re.compile(r'```.*?```', re.DOTALL)
    
    # Find all matches and return them
    blocks = pattern.findall(text)
    
    # Remove the triple backticks from each block
    blocks = [block[3:-3].strip() for block in blocks]
    
    return blocks