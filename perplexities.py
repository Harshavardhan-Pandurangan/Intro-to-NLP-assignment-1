import numpy as np
import re
import pprint as pp
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from math import log, exp
import sys

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

class NGram:

    def __init__(self):
        pass

    def setup(self, corpus_path, n):
        self.corpus_path = corpus_path
        self.n = n

        self.vocab = []
        self.tokens = []
        self.ngrams = {}

        self.smoothing = 'NA'
        self.probabilities = {}
        self.probabilities_gt = {}
        self.probabilities_i = {}

        tokenizer = Tokenizer(self.corpus_path)
        self.tokens = tokenizer.tokens

        i = 0
        while i < len(self.tokens):
            if len(self.tokens[i]) == 1 and (self.tokens[i][0] == ' ' or self.tokens[i][0] == ''):
                self.tokens.pop(i)
                i -= 1
            i += 1

        for i in range(self.n-1):
            for j in range(len(self.tokens)):
                self.tokens[j].insert(0, '<s>')
                self.tokens[j].append('</s>')

        np.random.seed(0)
        np.random.shuffle(self.tokens)
        self.train_tokens = self.tokens[:int(0.9*len(self.tokens))]
        self.test_tokens = self.tokens[int(0.9*len(self.tokens)):]

        self.vocab = list(set([word for token in self.tokens for word in token]))

    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            self.corpus = f.read()

    def train(self):
        seqs = {}
        for token in self.train_tokens:
            for i in range(len(token)-self.n+1):
                seq = tuple(token[i:i+self.n])
                if seq in seqs:
                    seqs[seq] += 1
                else:
                    seqs[seq] = 1

        for seq in seqs:
            if self.ngrams.get(seq[:-1]) is None:
                self.ngrams[seq[:-1]] = {}
            self.ngrams[seq[:-1]][seq[-1]] = seqs[seq]

        for seq in self.ngrams:
            total = sum(self.ngrams[seq].values())
            self.probabilities[seq] = {k: v/total for k, v in self.ngrams[seq].items()}

        self.smoothing = 'None'

    def save(self, file_path):
        if self.smoothing == 'NA':
            print('No model trained')
        elif self.smoothing == 'None':
            with open(file_path, 'w') as f:
                f.write(str(self.probabilities))
        elif self.smoothing == 'GT':
            with open(file_path, 'w') as f:
                f.write(str(self.probabilities_gt))
        elif self.smoothing == 'I':
            with open(file_path, 'w') as f:
                f.write(str(self.probabilities_i))

    def load(self, file_path):
        with open(file_path, 'r') as f:
            self.probabilities = eval(f.read())

    def perplexity(self, sentence):
        total_log_prob = 0
        total_tokens = 0

        probabilities = None
        if self.smoothing == 'NA':
            print('No model trained')
        elif self.smoothing == 'None':
            probabilities = self.probabilities
        elif self.smoothing == 'GT':
            probabilities = self.probabilities_gt
        elif self.smoothing == 'I':
            probs = []
            f_123, f_12, f_23, f_2, f_3 = 0, 0, 0, 0, 0
            for i in range(len(sentence)-2):
                token = sentence[i:i+3]
                if tuple(token) in self.ng3:
                    f_123 = self.ng3[tuple(token)]
                else:
                    f_123 = 0
                if (token[0], token[1]) in self.ng2:
                    f_12 = self.ng2[(token[0], token[1])]
                else:
                    f_12 = 0
                if (token[1], token[2]) in self.ng2:
                    f_23 = self.ng2[(token[1], token[2])]
                else:
                    f_23 = 0
                if token[1] in self.ng1:
                    f_2 = self.ng1[token[1]]
                else:
                    f_2 = 0
                if token[2] in self.ng1:
                    f_3 = self.ng1[token[2]]
                else:
                    f_3 = 0
                t1, t2, t3 = 0, 0, 0
                t1 = f_3 / sum(self.ng1.values())
                if f_2 != 0:
                    t2 = f_23 / f_2
                else:
                    t2 = 0
                if f_12 != 0:
                    t3 = f_123 / f_12
                else:
                    t3 = 0
                term = self.l3*t1 + self.l2*t2 + self.l1*t3

                if term != 0:
                    probs.append(term)
                else:
                    probs.append(10**-5)

            length = len(probs)
            r = 1
            for p in probs:
                r *= p

            if len(probs) == 0 or r == 0:
                return 0

            return r ** (-1 * (1 / length))

        flag = 0
        for i in range(len(sentence)-self.n):
            ngram = tuple(sentence[i:i+self.n-1])
            if ngram in probabilities:
                if sentence[i+self.n-1] in probabilities[ngram]:
                    total_log_prob += np.log(probabilities[ngram][sentence[i+self.n-1]])
                else:
                    total_log_prob += np.log(10**-5)
            else:
                total_log_prob += np.log(10**-5)
            total_tokens += 1
            flag += 1

        perplexity = np.exp(-total_log_prob/total_tokens)
        return perplexity

    def generate(self, prob_type='normal'):
        pass

    def evaluate(self):
        average_train = 0
        train_perp = []
        for i in range(len(self.train_tokens)):
            train_perp += [[self.train_tokens[i], self.perplexity(self.train_tokens[i])]]
            # average_train += self.perplexity(self.train_tokens[i])
            average_train += train_perp[-1][1]
        average_train /= len(self.train_tokens)

        average_test = 0
        test_perp = []
        for i in range(len(self.test_tokens)):
            test_perp += [[self.test_tokens[i], self.perplexity(self.test_tokens[i])]]
            # average_test += self.perplexity(self.test_tokens[i])
            average_test += test_perp[-1][1]
        average_test /= len(self.test_tokens)

        return average_train, average_test, train_perp, test_perp

    def good_turing(self):
        N_r = {}
        for seq in self.ngrams:
            for count in self.ngrams[seq].values():
                if count in N_r:
                    N_r[count] += 1
                else:
                    N_r[count] = 1

        N_r = {k: v for k, v in sorted(N_r.items(), key=lambda item: item[0])}

        probs_gt = {}
        for seq in self.ngrams:
            for token in self.ngrams[seq]:
                count = self.ngrams[seq][token]
                if count + 1 in N_r and count in N_r:
                    c_star = (count + 1) * (N_r[count + 1] / N_r[count])
                    probs_gt[seq + (token,)] = c_star / sum(self.ngrams[seq].values())
                else:
                    probs_gt[seq + (token,)] = 1 / len(self.vocab)

        self.smoothing = 'GT'

        self.probabilities_gt = {}
        for seq in probs_gt:
            if self.probabilities_gt.get(seq[:-1]) is None:
                self.probabilities_gt[seq[:-1]] = {}
            self.probabilities_gt[seq[:-1]][seq[-1]] = probs_gt[seq]

    def good_turning(self):
        Nr = {}
        for freq_dict in self.ngrams.values():
            freq = list(freq_dict.values())[0]
            if freq in Nr:
                Nr[freq] += 1
            else:
                Nr[freq] = 1

        Zr = {}
        Zr[1] = Nr[1]

        Nr_items = sorted(Nr.items())

        for i, thing in enumerate(Nr_items[1:-1], start=1):
            Zr[thing[0]] = (thing[1] * 2) / (Nr_items[i+1][0] - Nr_items[i-1][0])

        X = [[log(i[0])] for i in Zr.items()]
        y = [log(i[1]) for i in Zr.items()]

        md = LinearRegression()
        md.fit(X, y)

        self.Zr = Zr
        self.md = md

        self.probabilities_gt = {}

        for seq in self.ngrams:
            ngram = tuple(seq)
            r = list(self.ngrams[seq].values())[0]

            try:
                turing_estimate = (r + 1) * (Nr[r+1]) / Nr[r]
                self.probabilities_gt[ngram] = turing_estimate

                pred_logs = md.predict([[log(r+1)], [log(r)]])
                pred_estimate = (r + 1) * (exp(pred_logs[0])) / exp(pred_logs[1])

                variance = ((r + 1) ** 2) * (Nr[r+1] / (Nr[r] ** 2)) * (1 + ((Nr[r+1] / (Nr[r]))))

                GT_CONFIDENCE = 1.65

                if abs(turing_estimate - pred_estimate) < GT_CONFIDENCE * (variance ** 0.5):
                    self.probabilities_gt[ngram] = pred_estimate

            except KeyError:
                break

        self.smoothing = 'GT'

        with open('probabilities_gt.txt', 'w') as f:
            f.write(str(self.probabilities_gt))

    def interpolation(self, lambda_values=None):
        l1, l2, l3 = 0, 0, 0

        # self.ng1 will be the dict of all the tokens in the corpus
        self.ng1 = {}
        for token in self.tokens:
            for word in token:
                if word in self.ng1:
                    self.ng1[word] += 1
                else:
                    self.ng1[word] = 1

        # self.ng2 wil be the dict of all the bigrams in the corpus, with the first token being preceeded by <s> and the last token being followed by </s> for each sentence
        self.ng2 = {}
        for token in self.tokens:
            tup = ('<s>', token[0])
            if tup in self.ng2:
                self.ng2[tup] += 1
            else:
                self.ng2[tup] = 1
            for i in range(len(token)-1):
                tup = (token[i], token[i+1])
                if tup in self.ng2:
                    self.ng2[tup] += 1
                else:
                    self.ng2[tup] = 1
            tup = (token[-1], '</s>')
            if tup in self.ng2:
                self.ng2[tup] += 1
            else:
                self.ng2[tup] = 1

        # self.ng3 will be the dict of all the trigrams in the corpus, with the first two tokens being preceeded by <s> and the last token being followed by </s> for each sentence
        self.ng3 = {}
        for token in self.tokens:
            tup = ('<s>', '<s>', token[0])
            if tup in self.ng3:
                self.ng3[tup] += 1
            else:
                self.ng3[tup] = 1
            tup = ('<s>', token[0], token[1])
            if tup in self.ng3:
                self.ng3[tup] += 1
            else:
                self.ng3[tup] = 1
            for i in range(len(token)-2):
                tup = (token[i], token[i+1], token[i+2])
                if tup in self.ng3:
                    self.ng3[tup] += 1
                else:
                    self.ng3[tup] = 1
            tup = (token[-2], token[-1], '</s>')
            if tup in self.ng3:
                self.ng3[tup] += 1
            else:
                self.ng3[tup] = 1
            tup = (token[-1], '</s>', '</s>')
            if tup in self.ng3:
                self.ng3[tup] += 1
            else:
                self.ng3[tup] = 1

        N = len(self.ng1)

        for token in self.ng3:
            f_123, f_12, f_23, f_2, f_3 = 0, 0, 0, 0, 0

            if (token[0], token[1], token[2]) in self.ng3:
                f_123 = self.ng3[token]
            else:
                f_123 = 0

            if (token[0], token[1]) in self.ng2:
                f_12 = self.ng2[(token[0], token[1])]
            else:
                f_12 = 0

            if (token[1], token[2]) in self.ng2:
                f_23 = self.ng2[(token[1], token[2])]
            else:
                f_23 = 0

            if token[1] in self.ng1:
                f_2 = self.ng1[token[1]]
            else:
                f_2 = 0

            if token[2] in self.ng1:
                f_3 = self.ng1[token[2]]
            else:
                f_3 = 0

            t1, t2, t3 = 0, 0, 0
            if f_12 != 1:
                t1 = (f_123 - 1) / (f_12 - 1)
            else:
                t1 = 0
            if f_2 != 1:
                t2 = (f_23 - 1) / (f_2 - 1)
            else:
                t2 = 0
            t3 = (f_3 - 1) / (N - 1)

            m = max(t1, t2, t3)

            if m == 0:
                continue
            if m == t1:
                l1 += f_123
            elif m == t2:
                l2 += f_123
            else:
                l3 += f_123

        l1 /= (l1 + l2 + l3)
        l2 /= (l1 + l2 + l3)
        l3 /= (l1 + l2 + l3)

        self.l1, self.l2, self.l3 = l1, l2, l3

        # self.probabilities_i = {}

        # # the probabilities will be calculated using the lambda values
        # # the calculated values will be stored as dict of keys as n-1 gram tuple and each value will be a dict with key nth word and value probability
        # for token in self.ng3:
        #     total = self.ng3[token]
        #     self.probabilities_i[token[:-1]] = {}

        #     for word in self.ng1:
        #         unigram_prob = self.ng1[word] / sum(self.ng1.values())
        #         bigram_prob = self.ng2[token[1:]] / total
        #         trigram_prob = self.ng3[token] / total

        #         interpolated_prob = (l1 * unigram_prob) + (l2 * bigram_prob) + (l3 * trigram_prob)

        #         self.probabilities_i[token[:-1]][token[-1]] = interpolated_prob

        # print('Interpolation...Done')

        # with open('probabilities_i.txt', 'w') as f:
        #     f.write(str(self.probabilities_i))

        self.smoothing = 'I'

lm = 1
for corpus_path in ['Pride and Prejudice - Jane Austen.txt', 'Ulysses  James Joyce.txt']:
    for smoothing in ['gt', 'i']:
        ngram = NGram()
        ngram.setup(corpus_path, 3)
        ngram.train()
        if smoothing == 'gt':
            ngram.good_turing()
        else:
            ngram.interpolation()

        average_train, average_test, train_perp, test_perp = ngram.evaluate()

        with (open(f'2021111003_LM{lm}_train-perplexity.txt', 'w')) as f:
            f.write(str(average_train) + '\n')
            for item in train_perp:
                f.write(str(item[0]) + '\t' + str(item[1]) + '\n')

        with (open(f'2021111003_LM{lm}_test-perplexity.txt', 'w')) as f:
            f.write(str(average_test) + '\n')
            for item in test_perp:
                f.write(str(item[0]) + '\t' + str(item[1]) + '\n')

        lm += 1
