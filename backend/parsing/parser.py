from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os
import requests
import ast

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
