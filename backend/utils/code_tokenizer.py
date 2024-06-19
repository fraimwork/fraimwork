import pygments.lexers
import pygments.token
from transformers import AutoTokenizer
import pygments
from pygments import lex
from pygments.token import is_token_subtype
from collections import defaultdict

class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_index = defaultdict(lambda: 0, {token: index for index, token in enumerate(vocab)})
        self.index_to_token = defaultdict(lambda: '<UNK>', {index: token for index, token in enumerate(vocab)})
    
    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, token):
        return self.token_to_index[token]
    
    def __contains__(self, token):
        return token in self.token_to_index
    
    def get_token(self, index):
        return self.index_to_token[index]

class CodeTokenizer(Vocabulary):
    def __init__(self, lexer, framework_vocab=[], language_vocab=[]):
        self.lexer = lexer
        self.subword_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        subword_vocab = {token: index for index, token in enumerate(self.subword_tokenizer.get_vocab())}
        subword_vocab = [token for token, _ in sorted(subword_vocab.items(), key=lambda x: x[1])]
        self.vocab = subword_vocab + language_vocab + framework_vocab
        self.subword_vocab = Vocabulary(subword_vocab)
        self.framework_vocab = Vocabulary(framework_vocab)
        self.language_vocab = Vocabulary(language_vocab)
        self.language_offset = len(subword_vocab)
        self.framework_offset = self.language_offset + len(language_vocab)
        super().__init__(self.vocab)
    
    def is_further_lexable(self, token):
        token_type = token[0]
        critical_tokens = [pygments.token.Name, pygments.token.Literal, pygments.token.Comment]
        return any(is_token_subtype(token_type, critical_token) for critical_token in critical_tokens)
    
    def further_lexed(self, token):
        if is_token_subtype(token[0], pygments.token.Name):
            if token[1] in self.framework_vocab:
                return [token[1]]
        encoded_input = self.subword_tokenizer(token[1])
        # Decode the input IDs to get subwords (tokens)
        tokens = self.subword_tokenizer.convert_ids_to_tokens(encoded_input['input_ids'], skip_special_tokens=True)
        for token in tokens:
            yield token
    
    def tokenize(self, code):
        intitial_pass = list(lex(code, self.lexer))
        # After the initial pass, we will have a second pass where we BPE tokenize all custom-named tokens (comments, variable names, strings etc.)
        for token in intitial_pass:
            if self.is_further_lexable(token):
                for token in self.further_lexed(token):
                    yield token
            else:
                if is_token_subtype(token[0], pygments.token.Whitespace):
                    for char in token[1]:
                        yield char
                else:
                    yield token[1]
    
    def vectorize(self, code):
        return [index for index, _ in self.tokenize(code)]
    
    def lex_code(self, code):
        return lex(code, self.lexer)