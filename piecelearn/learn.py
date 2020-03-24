"""Learn wordpiece embeddings."""
import sentencepiece as spm

from gensim.models import Word2Vec
from itertools import tee


class Sentences(object):

    def __init__(self, generator_expression):
        self.generator_expression, self.g = tee(generator_expression)

    def __iter__(self):
        self.generator_expression, self.g = tee(self.generator_expression)
        return self

    def __next__(self):
        return next(self.g)


def train_spm(path_to_corpus, modelname, vocab_size=30000, **kwargs):
    opts = [f"--input={path_to_corpus}",
            f"--model_prefix={modelname}",
            f"--vocab_size={vocab_size}",
            f"--model_type=bpe"]
    spm.SentencePieceTrainer.Train(" ".join(opts))


def train_word2vec(lines,
                   spm_path,
                   out_path,
                   **kwargs):
    """Train a Word2Vec model on lines with an encoded model."""
    sp = spm.SentencePieceProcessor()
    # return value is boolean.
    sp.load(spm_path)

    lines = (sp.encode_as_pieces(x.strip()) for x in lines)

    s = Sentences(lines)
    model = Word2Vec(min_count=0, **kwargs)
    model.build_vocab(s)
    model.train(s,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    # Reorder vocabulary
    return model


def end_to_end(path_to_corpus,
               spm_model_name,
               vocab_size,
               spm_kwargs,
               word2vec_path,
               w2v_kwargs):
    """Train an spm model."""
    train_spm(path_to_corpus, spm_model_name, **spm_kwargs)
    train_word2vec(open(path_to_corpus),
                   f"{spm_model_name}.model",
                   word2vec_path,
                   **w2v_kwargs)
