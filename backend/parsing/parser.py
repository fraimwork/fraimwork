from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os
import requests
import ast


# We first create a base vocab list for Flutter and React Native
flutter_vocab = list(set([
    # dart constants
    "null", "true", "false", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # dart operator tokens
    "+", "-", "*", "/", "~", "%", "+", "-", "=", ">", "<", "&", "|", "!", "?"
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
    "Set", "Iterable", "Future", "Stream"
]))

react_native_vocab = list(set([
    # js constants
    "null", "true", "false", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # js operator tokens
    "+", "-", "*", "/", "~", "%", "+", "-", "=", ">", "<", "&", "|", "!", "?"
    # js keyword tokens
]))

def generate_flutter_vocabulary():
    # First, fetch the flutter project from GitHub
    git_repo = "https://github.com/flutter/flutter"
    # Then clone the repository into a tmp directory
    os.system(f"git clone {git_repo} tmp")
    # Next, extract the Dart files from the repository
    os.system("find tmp -name '*.dart' > dart_files.txt")
    # Finally, train the tokenizer on the Dart files
    dart_tokenizer = create_tokenizer('dart')

# Tokenizer setup (example with BPE tokenizer)
def create_tokenizer(language: str):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(vocab_size=30522, min_frequency=2, special_tokens=[
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    
    files = os.listdir(f"data/{language}")
    tokenizer.train(files, trainer)
    
    return tokenizer

dart_tokenizer = create_tokenizer('dart')
javascript_tokenizer = create_tokenizer('javascript')


import ast

def get_class_names(file_path):
    """Extracts class names from a Dart source file.

    Args:
        file_path: Path to the Dart source file.

    Returns:
        A list of class names found in the file.
    """
    with open(file_path, "r") as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

    return classes

# Example usage
file_path = "path/to/your/flutter/file.dart"
class_names = get_class_names(file_path)

print("Found classes:", class_names)
