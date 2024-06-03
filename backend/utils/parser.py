from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os
import requests
import ast

START_TOKEN = '<START>'
PADDING_TOKEN = '<PAD>'
END_TOKEN = '<END>'
END_OF_FILE_TOKEN = '<EOF>'
CHANGE_DIR_TOKEN = '<CD|>'

GENERICS = [START_TOKEN, PADDING_TOKEN, END_TOKEN, END_OF_FILE_TOKEN]


# We first create a base vocab list for Flutter and React Native
flutter_vocab = list(set([
    # dart constants
    "null", "true", "false", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # dart operator tokens
    "+", "-", "*", "/", "~", "%", "+", "-", "=", ">", "<", "&", "|", "!", "?", ":",
    # dart keyword tokens
    "abstract", "as", "assert", "async", "await", "break", "case", "catch", "class", "const",
    "continue", "covariant", "default", "deferred", "do", "dynamic", "else", "enum", "export",
    "extends", "extension", "external", "factory", "false", "final", "finally", "for", "Function",
    "get", "if", "implements", "import", "in", "interface", "is", "late", "library", "mixin",
    "new", "null", "on", "operator", "part", "rethrow", "return", "set", "show", "static",
    "super", "switch", "sync", "this", "throw", "true", "try", "typedef", "var", "void", "while",
    "with", "yield",
    # dart variable tokens
    "int", "double", "String", "bool", "List", "Map", "Set", "Iterable", "dynamic", "var"
    "final", "const", "static", "late", "final", "const", "static", "late",
    # dart function tokens
    "void", "main", "print", "assert", "import", "export", "extends", "implements", "super",
    "class", "interface", "abstract", "static", "final", "const", "var", "new", "this", "throw",
    "try", "catch", "finally", "return", "break", "continue", "if", "else", "switch", "case",
    "default", "while", "do", "for", "in", "assert", "with", "yield", "await", "async", "get",
    "set", "operator", "typedef", "typedef", "factory", "external", "library", "part", "import",
    "export", "is", "as", "extension", "on", "mixin", "with", "implements", "hide", "show",
    "native", "function", "dynamic", "void", "bool", "int", "double", "String", "List", "Map",
    "Set", "Iterable", "Future", "Stream",
    # misc
    " ", "\n", "\t", "(", ")", "{", "}", "[", "]", ",", ".", ";", ":", "=", "+", "-", "*", "/", "%"
])) + GENERICS

react_native_vocab = list(set([
    # js constants
    "null", "true", "false", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # js operator tokens
    "+", "-", "*", "/", "~", "%", "+", "-", "=", ">", "<", "&", "|", "!", "?", ":",
    # js keyword tokens
    'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 
    'delete', 'do', 'else', 'enum', 'export', 'extends', 'false', 'finally', 'for', 
    'function', 'if', 'import', 'in', 'instanceof', 'new', 'null', 'return', 'super', 
    'switch', 'this', 'throw', 'true', 'try', 'typeof', 'var', 'void', 'while', 
    'with', 'yield', 'await', 'implements', 'interface', 'let', 'package', 'private', 
    'protected', 'public', 'static', 'arguments', 'async', 'await', 'eval', 'of',
    # js variable tokens
    "let", "const", "var", "function", "class", "import",
    # js function tokens
    "function", "console", "log", "import", "export", "extends", "implements", "super",
    "class", "interface", "abstract", "static", "final", "const", "var", "new", "this", "throw",
    # misc
    " ", "\n", "\t", "(", ")", "{", "}", "[", "]", ",", ".", ";", ":", "=", "+", "-", "*", "/", "%"
])) + GENERICS