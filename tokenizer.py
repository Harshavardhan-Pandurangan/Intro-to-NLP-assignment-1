import re

class Tokenizer:
    def __init__(self, path):
        self.path = path
        self.tokens = []

        self.text = self.read_file()
        print(self.text)

        self.tokenize()

    def read_file(self):
        with open(self.path, 'r') as f:
            return f.read()

    def tokenize(self):
        pass