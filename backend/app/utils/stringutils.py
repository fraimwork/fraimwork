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

def find_most_similar_substring_naive(corpus, query, match_score=3, mismatch_penalty=-3, gap_penalty=-1):
    corpus_tokens = linewise_tokenize(corpus)
    query_tokens = linewise_tokenize(query)
    corpus_align, query_align, max_score, _= smith_waterman_tokens(corpus_tokens, query_tokens, match_score, mismatch_penalty, gap_penalty, processing=lambda x: x.lower().strip())
    return ''.join(corpus_align), max_score


db_kmers_cache = {}

def ktuple_matching(query, database, k):
    global db_kmers_cache
    if (''.join(database), k) in db_kmers_cache:
        database_kmers = db_kmers_cache[(''.join(database), k)]
    else:
        database_kmers = {}
        for i in range(len(database) - k + 1):
            kmer = ''.join(part.strip() for part in database[i:i+k])
            if kmer in database_kmers:
                database_kmers[kmer].append(i)
            else:
                database_kmers[kmer] = [i]
        db_kmers_cache[(''.join(database), k)] = database_kmers
    
    matches = []
    for i in range(len(query) - k + 1):
        kmer = ''.join(part.strip() for part in query[i:i+k])
        if kmer in database_kmers:
            for db_pos in database_kmers[kmer]:
                matches.append((i, db_pos))
    
    return matches

def find_top_n_diagonals(matches, n):
    diagonal_scores = {}
    for q_pos, db_pos in matches:
        diag = q_pos - db_pos
        if diag in diagonal_scores:
            diagonal_scores[diag] += 1
        else:
            diagonal_scores[diag] = 1
    
    sorted_diagonals = sorted(diagonal_scores.items(), key=lambda item: item[1], reverse=True)
    top_diagonals = [diag for diag, score in sorted_diagonals[:n]]
    
    return top_diagonals

def banded_dp(query, database, band_width, top_diagonals, match_score=3, mismatch_penalty=3, gap_penalty=1, processing=None):
    best_score = float('-inf')
    best_alignment = None
    
    for diag in top_diagonals:
        q_start = max(0, diag)
        db_start = max(0, -diag)
        q_end = min(len(query), len(database) + diag)
        db_end = min(len(database), len(query) - diag)
        
        dp = np.zeros((q_end - q_start + 1, db_end - db_start + 1))
        
        for i in range(1, q_end - q_start + 1):
            for j in range(max(1, i - band_width), min(db_end - db_start + 1, i + band_width)):
                q, d = query[q_start + i - 1], database[db_start + j - 1]
                if processing:
                    q = processing(q)
                    d = processing(d)
                multiplier = 1 + len(max(q, d, key=len))
                match = dp[i-1, j-1] + (match_score if q == d else -mismatch_penalty) * multiplier
                delete = dp[i-1, j] - gap_penalty * multiplier
                insert = dp[i, j-1] - gap_penalty * multiplier
                dp[i, j] = max(match, delete, insert)
        
        if dp[-1, -1] > best_score:
            best_score = dp[-1, -1]
            best_alignment = (q_start, db_start, q_end, db_end)
    
    return best_alignment, best_score

def fasta_algorithm(database, query, k=12, n=3, band_width=10, match_score=3, mismatch_penalty=3, gap_penalty=1):
    matches = ktuple_matching(query, database, k)
    top_diagonals = find_top_n_diagonals(matches, n)
    best_alignment, best_score = banded_dp(query, database, band_width, top_diagonals, match_score, mismatch_penalty, gap_penalty, processing=lambda x: x.lower().strip())
    if best_alignment is None:
        return None, float('-inf')
    q_start, db_start, q_end, db_end = best_alignment
    best_substring = database[db_start:db_end]
    
    return best_substring, best_score

def find_most_similar_substring(corpus, query):
    # 1. Tokenize the corpus and query into lines
    linewise_tokenized_corpus = linewise_tokenize(corpus)
    linewise_tokenized_substring = linewise_tokenize(query)

    # 2. Run the FASTA algorithm
    result, score = fasta_algorithm(linewise_tokenized_corpus, linewise_tokenized_substring, k=3, n=2, band_width=5, gap_penalty=0)
    if result is None:
        return None, float('-inf')

    # 3. Do a second pass with wordwise tokeization
    wordwise_tokenized_result = wordwise_tokenize(''.join(result))
    wordwise_tokenized_query = wordwise_tokenize(query)
    result, score = fasta_algorithm(wordwise_tokenized_result, wordwise_tokenized_query, k=3, n=3, band_width=3, match_score=2, mismatch_penalty=3, gap_penalty=5)
    if result is None:
        return None, float('-inf')
    return ''.join(result), score
