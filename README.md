# piecelearn

Byte Pair Encoding (BPE) is a compression technique that replaces commonly occurring substrings by atomic ids, which are sometimes called pieces. For example, if the substring `at` occurs often in your training data, we can save space by not representing `at` as two integer IDs every time it pops up, but instead allocating a single integer ID for the substring `at`. That way, we save a single ID every time we encounter the substring, and thus saves space at the cost of a larger dictionary.
BPE encoders can be employed to have a fixed vocabulary, speeding up training. For example, we can say we only want to represent the 30,000 most frequent pieces.
Unlike a word-based encoder, which doesn't know what to do with unseen words, a BPE-based encoder can be used to represent almost any word in the language on which the BPE encoder was trained.
For example, if we use a word-based encoder, and we have only seen `cat` during training, but not `cats`, our system would not know what to do. In a BPE-based system, the model can encode `cats` as `cats = cat + s`.
The only exception to this is the appearance of unknown letters. If the letter `x` never appears in your training data, the encoder will not be able to represent `catx` as `catx = cat + x`.

BPE embeddings can be a useful replacement for regular word-based word2vec embeddings, especially in noisy domains.
The script in this repository facilitates training BPE-based word embeddings on a corpus by training the BPE encoder and a word2vec model on the same set of texts.

## Requirements

* gensim
* sentencepiece
* numpy

## Usage

The script assumes your corpus has been preprocessed beforehand.
Things to consider:
* lowercasing
* removing punctuation
* removing numbers (setting them to 0)

Tokenization is not necessary.

Once you have a corpus, you can train a sentencepiece encoder and word2vec model like this:

```python
my_corpus = "file.txt"
num_pieces = 30000

spm_kwargs = {}
w2v_kwargs = {"sg": 1, "window": 15}

train(my_corpus,
      "spm_model_name",
      num_pieces,
      "w2v_path",
      spm_kwargs,
      w2v_kwargs)

```

The end result is a trained spm model and a trained word2vec model in .vec file format, both of which are _aligned_.

The spm model and embeddings can then be fed into [BPEmb](https://github.com/bheinzerling/bpemb), as follows:

```python
from bpemb import BPEmb
from bpemb.util import sentencepiece_load, load_word2vec_file

b = BPEmb(lang="en")
b.spm = sentencepiece_load("spm_model_name.model")
b.emb = load_word2vec_file("w2v_path")

s = b.embed("the dog flew over the fence")
print(s.shape)

```

## License

MIT

## Author

Stéphan Tulkens
