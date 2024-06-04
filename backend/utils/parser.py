from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os
import requests
import ast

START_TOKEN = '<START>'
PADDING_TOKEN = '<PAD>'
END_TOKEN = '<END>'
END_OF_FILE_TOKEN = '<EOF>'
CHANGE_DIR_TOKEN = '<CD|>'

GENERICS = [START_TOKEN, END_OF_FILE_TOKEN, END_TOKEN]


# We first create a base vocab list for Flutter and React Native
flutter_vocab = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@', ';',
                        '[', '\\', ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', '\n', '\t', '\r', '\x0b', '\x0c',
                        '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

react_native_vocab = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        '[', '\\', ']', '^', '_', '`', ';',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', '\n', '\t', '\r', '\x0b', '\x0c',
                        '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]