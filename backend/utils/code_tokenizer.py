import pygments.lexers
import pygments.token
from transformers import AutoTokenizer
import pygments
from pygments import lex
from pygments.token import is_token_subtype

class CodeTokenizer:
    def __init__(self, lexer):
        self.lexer = lexer
        self.subword_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.vocab = self.subword_tokenizer.get_vocab().keys()
    
    def is_further_lexable(self, token):
        token_type = token[0]
        critical_tokens = [pygments.token.Name, pygments.token.Literal, pygments.token.Comment]
        return any(is_token_subtype(token_type, critical_token) for critical_token in critical_tokens)
    
    def tokenize(self, code):
        intitial_pass = list(lex(code, self.lexer))
        # After the initial pass, we will have a second pass where we BPE tokenize all custom-named tokens (comments, variable names, strings etc.)
        for token in intitial_pass:
            if self.is_further_lexable(token):
                encoded_input = self.subword_tokenizer(token[1])
                # Decode the input IDs to get subwords (tokens)
                tokens = self.subword_tokenizer.convert_ids_to_tokens(encoded_input['input_ids'])
                for token in tokens:
                    yield token
            else:
                yield token[1]
    
    def lex_code(self, code):
        return lex(code, self.lexer)