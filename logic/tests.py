import pytest
from collections import defaultdict
import re

from utils.stringutils import (
    arr_from_sep_string, arr_from_numbered_list, extract_markdown_blocks,
    extract_snippet, markdown_to_dict, wordwise_tokenize, linewise_tokenize,
    string_edit_distance, raw_list_edit_distance, weighted_list_edit_distance, closest_substr
)

def test_arr_from_sep_string():
    assert arr_from_sep_string("a,b,c") == ['a', 'b', 'c']
    assert arr_from_sep_string("a, b, c", sep=',') == ['a', 'b', 'c']
    assert arr_from_sep_string("a|b|c", sep='|') == ['a', 'b', 'c']

def test_arr_from_numbered_list():
    assert arr_from_numbered_list("1 a\n2 b\n3 c") == ['a', 'b', 'c']
    assert arr_from_numbered_list("1. a\n2. b\n3. c") == ['a', 'b', 'c']
    assert arr_from_numbered_list("1 a\n2. b\n3 c") == ['a', 'b', 'c']

def test_extract_markdown_blocks():
    text = """Some text
    ```python
    print("Hello, World!")
    ```
    Some other text
    ```html
    <div>Hello, World!</div>
    ```"""
    assert extract_markdown_blocks(text) == ['print("Hello, World!")', '<div>Hello, World!</div>']

def test_extract_snippet():
    text = """Line 1
Line 2
Line 3
Line 4
Line 5"""
    assert extract_snippet(text, 'Line 3', 1) == 'Line 2\nLine 3\nLine 4'
    assert extract_snippet(text, 'Line 1', 2) == 'Line 1\nLine 2\nLine 3'
    assert extract_snippet(text, 'Line 5', 1) == 'Line 4\nLine 5'

def test_markdown_to_dict():
    markdown = """
# Header 1
Content under header 1

## Header 2
Content under header 2

# Header 3
Content under header 3
"""
    expected_result = {
        'header 1': 'Content under header 1',
        'header 2': 'Content under header 2',
        'header 3': 'Content under header 3'
    }
    assert markdown_to_dict(markdown) == expected_result

def test_wordwise_tokenize():
    assert wordwise_tokenize("Hello, world!") == ['Hello', ',', ' ', 'world']
    assert wordwise_tokenize("   Leading and trailing spaces   ") == ['Leading', ' ', 'and', ' ', 'trailing', ' ', 'spaces']

def test_linewise_tokenize():
    assert linewise_tokenize("1\n2\n3") == ['1', '\n', '2', '\n', '3']
    assert linewise_tokenize("Single line") == ['Single line']

def test_string_edit_distance():
    assert string_edit_distance("kitten", "sitting") == 3
    assert string_edit_distance("", "abc") == 3
    assert string_edit_distance("abc", "") == 3
    assert string_edit_distance("abc", "abc") == 0

def test_raw_list_edit_distance():
    assert raw_list_edit_distance(['a', 'b', 'c'], ['a', 'x', 'c']) == 1
    assert raw_list_edit_distance(['a'], ['a', 'b']) == 1
    assert raw_list_edit_distance([], ['a', 'b']) == 2

def test_weighted_list_edit_distance():
    assert weighted_list_edit_distance(['a', 'b', 'c'], ['a', 'x', 'c']) == 1.0
    assert weighted_list_edit_distance(['a'], ['a', 'b']) == 1.0
    assert weighted_list_edit_distance([], ['a', 'b']) == 2.0
    assert weighted_list_edit_distance(['a', 'and good day to you', 'c'], ['a', 'and a good day to you', 'c']) <= 3.0

def test_closest_substr():
    main = "This is a test string.\nLet's see how it works.\nTesting one two three."
    sub = "see how it works"
    assert closest_substr(main, sub)[0] == "Let's see how it works."
    main = "This is a test string with multiple newlines\n\n\nLet's see how it works.\n\n Testing one two three."
    sub = "with multiple newlines\n\n...\n. Let's see how it works"
    assert closest_substr(main, sub)[0] == "This is a test string with multiple newlines\n\n\nLet's see how it works."

if __name__ == '__main__':
    pytest.main()
