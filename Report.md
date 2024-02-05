# INLP - Assignment 1 - Report

## 2021111003

### Analysis

-   The perplexities have been found in the python notebooks and the results are as follows:
    -   For the regular model, the perplexities range from 2 to 10 for train and in thousands for test.
    -   For good turing, it ranges from 5 to 100 for train and in thousands for test.
    -   The good turing test perplexities are lower than the regular model test perplexities.
    -   For linear interpolation, it ranges from 20 to 50 for test set.

Linear interpolation and Good-Turing smoothing impact perplexity differently in n-gram models. Linear interpolation, which combines probabilities from lower and higher-order n-grams, often helps decrease perplexity by improving the model's adaptability to different n-gram contexts. On the other hand, Good-Turing smoothing tends to decrease perplexity by providing more accurate estimates for unseen or rare n-grams, enhancing the model's overall predictive performance. Both techniques play crucial roles in addressing the challenges posed by unseen events and contribute to the optimization of n-gram language models.

### Generator

-   The generator has been tested with generation of various sentences and k values. The results are as follows:
    -   For case of normal model, the generated tokens are most frequent tokens in the corpus, in the respective n-1 gram context.
        = For case of OOD tokens, the code generates with restpect to matching n-1 gram context, with most matching tokens with the given token. So n-1 grams that match a lot with the given token sequence are given more weightage.
    -   For case of good turing and linear interpolation, the generated tokens are most frequent tokens in the corpus, in the respective n-1 gram context, and smoothed out with the respective smoothing technique in terms of the probability.
