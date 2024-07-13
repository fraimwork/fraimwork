import re

def arr_from_sep_string(string: str, sep=","):
    return [x.strip() for x in string.split(sep)]

def extract_markdown_blocks(text):
    """
    Extract markdown blocks from a string.

    Parameters:
    text (str): The input string containing markdown blocks.

    Returns:
    list: A list of strings, each representing a markdown block.
    """
    pattern = re.compile(r'```.*?```', re.DOTALL)
    blocks = pattern.findall(text)
    blocks = [block[3:-3].strip() for block in blocks]
    return blocks

def remove_indents(string: str):
    """
    Remove leading whitespace from each line of a string.

    Parameters:
    string (str): The input string.

    Returns:
    str: The string with leading whitespace removed from each line.
    """
    return "\n".join([line.lstrip() for line in string.split("\n")])