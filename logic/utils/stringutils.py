import re, difflib
from collections import defaultdict
import numpy as np
from functools import lru_cache

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
    """Tokenizes a sequence into words, punctuation, and whitespace

    Args:
        text: The input text string.

    Returns:
        A list of tokens.
    """

    # Split the text into words and whitespace
    tokens = re.findall(r'([a-zA-Z]+|\s+|[^a-zA-Z\s]+)', text)

    return tokens

def linewise_tokenize(text: str):
    """Tokenizes a sequence into lines and \n.

    Args:
        text: The input text string.

    Returns:
        A list of tokens.
    """
    # Split the text into lines and newlines
    return text.splitlines(keepends=True)

@lru_cache(maxsize=None)
def _edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j],      # Remove
                    dp[i - 1][j - 1]   # Replace
                )
    return dp[m][n]


edit_distance_cache = {}
def edit_distance(list1, list2):
    global edit_distance_cache
    if (tuple(list1), tuple(list2)) in edit_distance_cache:
        return edit_distance_cache[(tuple(list1), tuple(list2))]
    m, n = len(list1), len(list2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = sum(_edit_distance("", list2[k]) for k in range(j))
            elif j == 0:
                dp[i][j] = sum(_edit_distance(list1[k], "") for k in range(i))
            else:
                cost_replace = _edit_distance(list1[i - 1], list2[j - 1])
                dp[i][j] = min(dp[i - 1][j] + _edit_distance(list1[i - 1], ""),  # Remove
                               dp[i][j - 1] + _edit_distance("", list2[j - 1]),  # Insert
                               dp[i - 1][j - 1] + cost_replace)                # Replace
    edit_distance_cache[(tuple(list1), tuple(list2))] = dp[m][n]
    return dp[m][n]

def smith_waterman_tokens(key, query, match_score=3, mismatch_penalty=-3, gap_penalty=-2, processing=None, diagonal=None, key_token_depth=None, query_token_depth=None):
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
            align1.insert(0, '-')
            j -= 1

    return align1, align2, max_score, scoring_matrix

def find_most_similar_substring_naive(corpus, query, match_score=3, mismatch_penalty=-3, gap_penalty=-1):
    corpus_tokens = linewise_tokenize(corpus)
    query_tokens = linewise_tokenize(query)
    corpus_align, query_align, max_score, _= smith_waterman_tokens(corpus_tokens, query_tokens, match_score, mismatch_penalty, gap_penalty, processing=lambda x: x.lower().strip())
    return ''.join(corpus_align), max_score

def score(a, b):
    max_len = max(len(''.join(a)), len(''.join(b)))
    if max_len == 0: return 1
    diff = edit_distance(a, b)
    return 1 - (diff / max_len)

def weighted_score(a, b):
    return score(a, b) * np.mean([len(a), len(b)])

db_kmers_cache = {}

def ktuple_matching(query, database, k, part_processing=None):
    global db_kmers_cache
    if not part_processing:
        part_processing = lambda x: x
    if (''.join(database), k) in db_kmers_cache:
        database_kmers = db_kmers_cache[(''.join(database), k)]
    else:
        database_kmers = defaultdict(list)
        for i in range(len(database) - k + 1):
            kmer = ''.join(part_processing(part) for part in database[i:i+k])
            if ''.join(kmer) ==  '': continue
            database_kmers[kmer].append(i)
        db_kmers_cache[(''.join(database), k)] = database_kmers
    
    matches = []
    for i in range(len(query) - k + 1):
        kmer = ''.join(part_processing(part) for part in query[i:i+k])
        if ''.join(kmer) == '': continue
        if kmer in database_kmers:
            for db_pos in database_kmers[kmer]:
                matches.append((i, db_pos))
    return matches

def find_top_diagonal(matches, query, database, k):
    diagonal_scores = defaultdict(int)
    for q_pos, db_pos in matches:
        diag = q_pos - db_pos
        diagonal_scores[diag] += np.mean([weighted_score(q, db) for q, db in zip(query[q_pos:q_pos+k], database[db_pos:db_pos+k])])
    top_diagonal = max(diagonal_scores.items(), key=lambda item: item[1], default=(None, 0))
    return top_diagonal[0]

def smith_waterman_diagonal(
        query,
        database,
        match_score=3,
        mismatch_penalty=3,
        gap_penalty=2,
        processing=None,
        tolerance=0.75,
        diag=0,
        band_width=5,
        db_token_depth_roc=None,
        query_token_depth_roc=None
    ):
    m, n = len(query), len(database)
    
    scoring_matrix, traceback_matrix = defaultdict(int), defaultdict(int)
    if diag >= 0:
        start_row = 1
        start_col = -diag
    else:
        start_row = diag
        start_col = 1

    max_score = 0
    max_pos = (0, 0)
    covered = []
    while start_row <= m and start_col <= n:
        for offset in range(-band_width, band_width + 1):
            row, col = start_row, start_col + offset
            if not (0 < row <= m and 0 < col <= n): continue
            q, d = query[row-1], database[col-1]
            covered.append((row - 1, col - 1))
            if processing:
                q, d = processing(q), processing(d)
            multiplier = 1 + np.mean([len(q), len(d)]) / 2
            match = scoring_matrix[(row-1, col-1)] + (match_score if score(q, d) >= tolerance else -mismatch_penalty) * multiplier
            insert = scoring_matrix[(row, col-1)] - gap_penalty * multiplier
            scoring_matrix[(row, col)] = max(match, insert)

            if scoring_matrix[(row, col)] == match:
                traceback_matrix[(row, col)] = 1
            elif scoring_matrix[(row, col)] == insert:
                traceback_matrix[(row, col)] = 3
            
            if scoring_matrix[(row, col)] >= max_score:
                max_score = scoring_matrix[(row, col)]
                max_pos = (row, col)
        start_row += 1
        start_col += 1

    # Traceback to get the optimal alignment
    align1, align2 = [], []
    i, j = max_pos
    while scoring_matrix[(i, j)] > 0:
        q, d = query[i - 1], database[j - 1]
        if traceback_matrix[(i, j)] == 1:
            align1.insert(0, q)
            align2.insert(0, d)
            i -= 1
            j -= 1
        elif traceback_matrix[(i, j)] == 3:
            align2.insert(0, d)
            align1.insert(0, '-')
            j -= 1
    return align2, max_score, covered

def fasta_algorithm(database, query, k=4, n=3, band_width=5, match_score=3, mismatch_penalty=3, gap_penalty=1, match_processing=None, dp_processing=None, db_token_depth_roc=None, query_token_depth_roc=None):
    matches = ktuple_matching(query, database, k, match_processing)
    top_diagonal = find_top_diagonal(matches, query, database, k)
    if top_diagonal is None: return None, float('-inf')
    best_alignment, best_score, covered = smith_waterman_diagonal(
        query,
        database,
        match_score=match_score,
        mismatch_penalty=mismatch_penalty,
        gap_penalty=gap_penalty,
        processing=dp_processing,
        diag=top_diagonal,
        band_width=band_width,
        db_token_depth_roc=db_token_depth_roc,
        query_token_depth_roc=query_token_depth_roc
    )
    if best_alignment is None:
        return None, float('-inf')
    return best_alignment, best_score

def fuzzy_find(query, database):
    # 1. First pass with linewise tokenization
    linewise_tokenized_database = linewise_tokenize(database)
    linewise_tokenized_query = linewise_tokenize(query)

    result, score = fasta_algorithm(
        linewise_tokenized_database,
        linewise_tokenized_query,
        k=1,
        n=1,
        band_width=10,
        match_score=3,
        mismatch_penalty=3,
        gap_penalty=0.1,
        match_processing=lambda x: x.strip().replace(' ', ''),
        dp_processing=lambda x: wordwise_tokenize(x.strip().replace(' ', '')),
    )

    # 2. Second pass with wordwise tokenization
    worwise_tokenized_database = wordwise_tokenize(''.join(result)) if result is not None else wordwise_tokenize(database)
    wordwise_tokenized_query = wordwise_tokenize(query)

    result, score = fasta_algorithm(
        worwise_tokenized_database,
        wordwise_tokenized_query,
        k=max(1, min(3, len(wordwise_tokenized_query) - 1, len(worwise_tokenized_database) - 1)),
        n=1,
        band_width=18,
        match_score=5,
        mismatch_penalty=3,
        gap_penalty=0.3,
        match_processing=lambda x: x.strip(),
        dp_processing=lambda x: x.strip(),
    )

    if result is None:
        return None, float('-inf')
    return ''.join(result), score

def get_diffs(a, b):
    diffs = [diff for diff in difflib.ndiff(a.splitlines(), b.splitlines())]
    diffs_groups = []
    current_group = []
    for diff in diffs:
        if diff.startswith(' '):
            if current_group:
                diffs_groups.append(current_group)
                current_group = []
        elif diff.startswith('-') or diff.startswith('+'):
            current_group.append(diff)
    diffs = []
    for group in diffs_groups:
        # Turn group into an array of tuples (find, replace)
        find = '\n'.join([line[1:] for line in group if line.startswith('-')])
        replace = '\n'.join([line[1:] for line in group if line.startswith('+')])
        diffs.append((find, replace))
    return diffs

def parse_diff(diff_string):
    lines = diff_string.splitlines()
    groups = []
    current_group = []
    in_edit = False

    for line in lines:
        if line.startswith('+') or line.startswith('-'):
            if not in_edit and current_group:
                groups.append(current_group)
                current_group = []
            current_group.append(line)
            in_edit = True
        else:
            if in_edit and current_group:
                groups.append(current_group)
                current_group = []
            current_group.append(line)
            in_edit = False
    
    if current_group:
        groups.append(current_group)
    
    merged_groups = []
    for group in groups:
        if not (group[0].startswith('+') or group[0].startswith('-')):
            merged_groups.append('\n'.join(group))
        else:
            delete, insert = '', ''
            for line in group:
                if line.startswith('-'):
                    delete += line[1:] + '\n'
                elif line.startswith('+'):
                    insert += line[1:] + '\n'
            merged_groups.append((delete, insert))

    return merged_groups

def skwonk(database: str, diff: str):
    hunks = re.split(r"@@.*?@@", diff)
    i = 0
    for hunk in hunks:
        original = ''.join([line for line in hunk.splitlines(keepends=True) if not line.startswith('+')])
        # 1. Zone in on the string we are going to be editing
        zone, _ = fuzzy_find(original, database[i:])
        i = database.find(zone) + len(zone)
        original_zone = zone
        if zone is None:
            return database
        
        # 2. Parse the diff into groups
        diff_groups = parse_diff(hunk)
        # 3. Find and replace
        for i in range(len(diff_groups)):
            group = diff_groups[i]
            if not isinstance(group, tuple): continue
            find, replace = group
            if find == '':
                # Insertion
                prior_context = fuzzy_find(diff_groups[i - 1], zone)[0] if i > 0 else None
                further_context = fuzzy_find(diff_groups[i + 1], zone)[0] if i < len(diff_groups) - 1 else None
                insertion_index = zone.find(prior_context) + len(prior_context) if prior_context else None
                if not insertion_index:
                    insertion_index = zone.find(further_context) if further_context else 0
                zone = zone[:insertion_index] + '\n' + replace + zone[insertion_index:]
            else:
                # Fuzzy find and replace
                fuzz, _ = fuzzy_find(find, zone)
                if fuzz is None: continue
                zone = zone.replace(fuzz, replace, 1)
        database = database.replace(original_zone, zone, 1)
    return database

def find_most_similar_file_name(files, query):
    return max(
        (file for file in files),
        key=lambda file: fasta_algorithm(file, query, k=6)[1]
    )

def compute_nested_levels(code_lines: list[str], indent_is_relevant: bool = True) -> list[int]:
    nested_levels = []
    brace_level = 0
    paren_level = 0
    indent_level = 0
    current_level = 0
    prev_indent_spaces = 0

    for line in code_lines:
        if line.strip() == '':
            nested_levels.append(current_level)
            continue
        stripped_line = line.rstrip()
        # Update the brace and parenthesis levels for the next line
        brace_level += stripped_line.count('{')
        brace_level -= stripped_line.count('}')
        paren_level += stripped_line.count('(')
        paren_level -= stripped_line.count(')')
        if indent_is_relevant:
            leading_spaces = len(line) - len(line.lstrip())
            indent_level += np.sign(leading_spaces - prev_indent_spaces)
            prev_indent_spaces = leading_spaces
        # Make sure levels do not go negative
        if brace_level < 0:
            brace_level = 0
        if paren_level < 0:
            paren_level = 0
        
        # Calculate the current line's nested level
        current_level = brace_level + paren_level + indent_level
        nested_levels.append(current_level)
    return nested_levels