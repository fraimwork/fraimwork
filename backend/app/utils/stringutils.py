import re
from collections import defaultdict
import numpy as np

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

import re

def wordwise_tokenize(text):
    """Tokenizes a sequence into words and whitespace

    Args:
        text: The input text string.

    Returns:
        A list of tokens.
    """

    # Split the text into words and whitespace
    tokens = re.findall(r'\w+|\s+|[^\w\s]', text)

    return tokens

def linewise_tokenize(text):
    """Tokenizes a sequence into lines and \n.

    Args:
        text: The input text string.

    Returns:
        A list of tokens.
    """
    # Split the text into lines and newlines
    tokens = re.split(r'(\n)', text)
    return tokens

def string_edit_distance(a: str, b: str):
    """
    Compute the Levenshtein distance between two strings.

    Parameters:
    a (str): The first string.
    b (str): The second string.

    Returns:
    int: The Levenshtein distance between the two strings.
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def smith_waterman_tokens(key, query, match_score=3, mismatch_penalty=-3, gap_penalty=-2, processing=None):
    # Initialize the scoring matrix and the traceback matrix
    rows = len(key) + 1
    cols = len(query) + 1
    scoring_matrix = np.zeros((rows, cols), dtype=int)
    traceback_matrix = np.zeros((rows, cols), dtype=int)

    max_score = 0
    max_pos = (0, 0)

    # Fill the scoring matrix and the traceback matrix
    for i in range(1, rows):
        for j in range(1, cols):
            t1, t2 = key[i-1], query[j-1]
            if processing:
                t1 = processing(t1)
                t2 = processing(t2)
            multiplier = 1 + len(max(t1, t2, key=len))
            match = scoring_matrix[i-1, j-1] + (match_score * multiplier if t1 == t2 else mismatch_penalty * multiplier)
            delete = scoring_matrix[i-1, j] + gap_penalty * multiplier
            insert = scoring_matrix[i, j-1] + gap_penalty * multiplier
            scoring_matrix[i, j] = max(0, match, delete, insert)

            if scoring_matrix[i, j] == match:
                traceback_matrix[i, j] = 1  # Diagonal
            elif scoring_matrix[i, j] == delete:
                traceback_matrix[i, j] = 2  # Up
            elif scoring_matrix[i, j] == insert:
                traceback_matrix[i, j] = 3  # Left

            if scoring_matrix[i, j] >= max_score:
                max_score = scoring_matrix[i, j]
                max_pos = (i, j)

    # Traceback to get the optimal alignment
    align1, align2 = [], []
    i, j = max_pos
    while scoring_matrix[i, j] != 0:
        if traceback_matrix[i, j] == 1:
            align1.insert(0, key[i-1])
            align2.insert(0, query[j-1])
            i -= 1
            j -= 1
        elif traceback_matrix[i, j] == 2:
            align1.insert(0, key[i-1])
            align2.insert(0, '-')
            i -= 1
        elif traceback_matrix[i, j] == 3:
            align2.insert(0, query[j-1])
            j -= 1

    return align1, align2, max_score, scoring_matrix

def find_most_similar_substring(corpus, query, match_score=3, mismatch_penalty=-3, gap_penalty=-1):
    corpus_tokens = linewise_tokenize(corpus)
    query_tokens = linewise_tokenize(query)
    corpus_align, query_align, max_score, _= smith_waterman_tokens(corpus_tokens, query_tokens, match_score, mismatch_penalty, gap_penalty, processing=lambda x: x.lower().strip())
    return ''.join(corpus_align), max_score
