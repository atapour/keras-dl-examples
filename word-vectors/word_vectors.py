# -*- coding: utf-8 -*-
"""
# Python code for a demonstration of Word Embeddings (Word2Vec)
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@newcastle.ac.uk

This notebook will provide a simple example that loads pre-trained word embeddings and performs a few arithmetic operations.

This is a code demonstration for CSC8637: Deep Learning module, Lecture 09: Time-Series Data: Natural Language Processing.

Copyright (c) 2021 Amir Atapour-Abarghouei, UK.

License : LGPL - http://www.gnu.org/licenses/lgpl.html

Here, we want to see what Word2Vec can do. There has been a lot of hype about this and many of you may already be familiar with the word embedding arithmetic.

We will download the Word2Vec model trained on part of the Google News dataset, covering approximately 3 million words and phrases. While this model is not the strongest, it can still take hours to train. In this demo, we will use a pre-trained version that can be downloaded and loaded within minutes.

You might want to play around with a more robust online demo trained on the entire Google News dataset.

The demo can be found here: https://rare-technologies.com/word2vec-tutorial/#app

To make things easier, we use the [Gensim library](https://pypi.org/project/gensim/), which is used for unsupervised topic modelling and natural language processing.

Let's get to it! First, we import what we need from the Gensim library.
"""

# Required imports:
import gensim.downloader as api
import gensim
import sys

# All python warnings are ignored here:
import warnings
warnings.filterwarnings(action='ignore')

# Printing the versions of Python and Gensim:
print('Python version:', sys.version)
print('Gensim version:', gensim.__version__)

"""We should download the pre-trained word vectors and check how many words are available:"""

# Download and load the Word Vectors (wv):
wv = api.load('word2vec-google-news-300')
print(f'{len(wv.vocab.keys())} words are available.')

"""The [`most_similar`](https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.most_similar.html) command will find the top-N most similar words within our vocabulary."""

wv.most_similar('banana')

wv.most_similar('dog')

"""We can try to find **dissimilar** words using the `negative` argument. You will see that this normally won't give us any meaningful information, but it can enable word arithmetics."""

wv.most_similar(negative='rich')

"""We can try doing operations like `woman - man + king `, which essentially gives us "*Man is to King as Woman is to ...*""""

result = wv.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.2f}".format(*result[0]))

"""We can write a funcion `analogy()` that does the above operation for us:"""

# A function that performs the word vector arithmetics:
def analogy(x1, x2, y1):
    result = wv.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]

"""Now, let's try it out:"""

analogy('father', 'mother', 'brother')

analogy('father', 'son', 'mother')

analogy('britain', 'british', 'australia')

analogy('flower', 'petal', 'tree')

analogy('clean', 'cleaner', 'far')

analogy('good', 'well', 'quick')

analogy('good', 'fantastic', 'ugly')

"""The [`doesnt_match`](hhttps://tedboy.github.io/nlps/generated/generated/gensim.models.Doc2Vec.doesnt_match.html) function can tell us which word from a given list doesn’t go with the others."""

print(wv.doesnt_match("table garden chair sofa".split()))

print(wv.doesnt_match("virus pandemic lockdown party".split()))

"""We should always keep in mind that things don't always work out the way we want them to. The vectors we are using here have been trained on a relatively small dataset, but even stronger word embeddings trained on massive corpora of text still suffer from significant issues, mostly due to the bias structurally inherent within the language and the cultures associated with it.

We will talk about this *algorithmic bias* in a future lecture.
"""

analogy('britain', 'british', 'sweden')

analogy('man', 'schoolteacher', 'woman')