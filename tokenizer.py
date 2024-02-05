import numpy as np
import re
import pprint as pp
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from math import log, exp

# Define a Tokenizer class
class Tokenizer:
    # Initialize the Tokenizer object with optional file path or sentence
    def __init__(self, path='None', sentence='None'):
        # Set the file path and sentence attributes
        self.path = path
        self.sentence = sentence
        self.tokens = []  # List to store processed tokens

        # Define regular expressions for various patterns
        self.sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\n)\s')
        self.newline_pattern = re.compile(r'\n')
        self.mention_pattern = re.compile(r'@(\w+)')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.mail_pattern = re.compile(r'(\S+@\S+\.\w+)')
        self.url_pattern = re.compile(r'\S+\.\w+')
        self.num_pattern = re.compile(r'(?<!\w)(\d+\.?\d+|\.\d+)(?!\w)')
        self.punctuation_spacing_pattern1 = re.compile(r'(?<!\s)([^\w\s])(?!\S)')
        self.punctuation_spacing_pattern2 = re.compile(r'(?<!\S)([^\w\s])(?!\s)')

        # Read the file if a path is provided, or use the given sentence
        if self.path != 'None':
            self.text = self.readFile()
        elif self.sentence != 'None':
            self.text = self.sentence

        # Preprocess the text by detecting and replacing specific patterns with tags
        self.detectAndReplaceWithTags()

        # Tokenize the text into sentences, separate characters, and fix tags
        self.separateChars()
        self.fixTags()

        # Uncomment the line below to print the resulting tokens
        # pp.pprint(self.tokens)

    # Read the content of the file specified by the path
    def readFile(self):
        with open(self.path, 'r') as f:
            return f.read()

    # Detect and replace newlines with spaces in a sentence
    def newlineDetector(self, sentence):
        return re.sub(self.newline_pattern, ' ', sentence)

    # Detect and replace email addresses with '<MAILID>' in a sentence
    def mailDetector(self, sentence):
        return re.sub(self.mail_pattern, '<MAILID>', sentence)

    # Detect and replace mentions with '<MENTION>' in a sentence
    def mentionDetector(self, sentence):
        return re.sub(self.mention_pattern, '<MENTION>', sentence)

    # Detect and replace hashtags with '<HASHTAG>' in a sentence
    def hashtagDetector(self, sentence):
        return re.sub(self.hashtag_pattern, '<HASHTAG>', sentence)

    # Detect and replace URLs with '<URL>' in a sentence
    def urlDetector(self, sentence):
        return re.sub(self.url_pattern, '<URL>', sentence)

    # Detect and replace numbers with '<NUM>' in a sentence
    def numDetector(self, sentence):
        return re.sub(self.num_pattern, '<NUM>', sentence)

    # Tokenize the text into sentences using a defined pattern
    def sentenceTokenizer(self, text):
        return re.split(self.sentence_pattern, text)

    # Process the entire text by applying various detectors to each sentence
    def detectAndReplaceWithTags(self):
        sentences = self.sentenceTokenizer(self.text)

        for sentence in sentences:
            sentence = self.newlineDetector(sentence)
            sentence = self.mailDetector(sentence)
            sentence = self.mentionDetector(sentence)
            sentence = self.hashtagDetector(sentence)
            sentence = self.numDetector(sentence)
            sentence = self.urlDetector(sentence)

            # Append the processed sentence to the tokens list
            self.tokens.append(sentence)

    # Separate characters within each token to add spacing around punctuation
    def separateChars(self):
        for i in range(len(self.tokens)):
            length = 0
            # Repeat until no further changes are made to the token
            while length != len(self.tokens[i]):
                length = len(self.tokens[i])
                # Add space around punctuation using defined patterns
                self.tokens[i] = re.sub(self.punctuation_spacing_pattern1, r' \1', self.tokens[i])
                self.tokens[i] = re.sub(self.punctuation_spacing_pattern2, r'\1 ', self.tokens[i])
            # Remove extra whitespaces and split the token into a list of words
            self.tokens[i] = re.sub(r'\s+', ' ', self.tokens[i])
            self.tokens[i] = self.tokens[i].strip()
            self.tokens[i] = self.tokens[i].split(' ')

    # Fix tags in the tokens list, replacing specific sequences with corresponding tags
    def fixTags(self):
        for i in range(len(self.tokens)):
            j = 0
            # Iterate through each element in the token
            while j < len(self.tokens[i]):
                # Check for specific tag patterns and replace accordingly
                if self.tokens[i][j] == '>' and self.tokens[i][j-2] == '<':
                    if self.tokens[i][j-1] == 'MAILID':
                        self.tokens[i][j-2] = '<MAILID>'
                        self.tokens[i].pop(j-1)
                        self.tokens[i].pop(j-1)
                    elif self.tokens[i][j-1] == 'MENTION':
                        self.tokens[i][j-2] = '<MENTION>'
                        self.tokens[i].pop(j-1)
                        self.tokens[i].pop(j-1)
                    elif self.tokens[i][j-1] == 'HASHTAG':
                        self.tokens[i][j-2] = '<HASHTAG>'
                        self.tokens[i].pop(j-1)
                        self.tokens[i].pop(j-1)
                    elif self.tokens[i][j-1] == 'URL':
                        self.tokens[i][j-2] = '<URL>'
                        self.tokens[i].pop(j-1)
                        self.tokens[i].pop(j-1)
                    elif self.tokens[i][j-1] == 'NUM':
                        self.tokens[i][j-2] = '<NUM>'
                        self.tokens[i].pop(j-1)
                        self.tokens[i].pop(j-1)
                j += 1

# Take user input for text
input_text = input('Enter the text: ')

# Create a Tokenizer object with the user-provided text
tokenizer = Tokenizer(sentence=input_text)

# Write the resulting tokens to a file
with open('tokens.txt', 'w') as f:
    f.write(str(tokenizer.tokens))
