"""Learn wordpiece embeddings."""
import sentencepiece as spm
import logging
import numpy as np

from gensim.models import Word2Vec
from itertools import tee


NO_OPT = {"input", "model_prefix", "vocab_size", "model_type"}
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Sentences(object):
    """
    An iterator over an arbitrary generator expression.

    adapted from here:
    https://jacopofarina.eu/posts/gensim-generator-is-not-iterator/
    """

    def __init__(self, generator_expression):
        """
        parameters
        ----------
        generator_expression : generator
            A generator.

        """
        self.generator_expression, self.g = tee(generator_expression)

    def __iter__(self):
        """Start from a new iterator over the generator."""
        self.generator_expression, self.g = tee(self.generator_expression)
        return self

    def __next__(self):
        return next(self.g)


def reorder_embeddings(sp, wv):
    """reorder embeddings."""
    vecs = wv.vectors
    # Create new random vectors with same mean and std
    mean, std = vecs.mean(0), vecs.std(0)
    new_vecs = np.random.normal(mean, std, size=(len(sp), vecs.shape[1]))
    new_indices = np.asarray([sp.piece_to_id(x) for x in wv.index2word])
    mask = new_indices != 0
    masked = new_indices[mask]
    assert np.all(np.unique(masked, return_counts=True)[1] == 1)
    new_vecs[masked] = vecs[mask]
    return [sp.id_to_piece(x) for x in range(len(sp))], new_vecs


def train_spm(path_to_corpus, modelname, vocab_size=30000, **kwargs):
    """
    Train an SPM model.

    This simply calls the `spm_train` command, with all flags passed in as a
    string. This function thus simply assembles such a string from the
    path_to_corpus, modelname, vocab_size arguments, and any other arguments
    the user passes in through kwargs.
    Make sure all the kwargs that are passed in match the names of the
    arguments of spm_train exactly, no checking is done.

    Parameters
    ----------
    path_to_corpus : str
        The path to the input corpus.
    modelname : str
        The name of the spm model.
    vocab_size : int, default 30000
        The number of wordpieces to make

    """
    intersection = set(kwargs.keys()) & NO_OPT
    if intersection:
        raise ValueError(f"{intersection} can not be assigned via kwargs")

    opts = [f"--input={path_to_corpus}",
            f"--model_prefix={modelname}",
            f"--vocab_size={vocab_size}",
            f"--model_type=bpe"]
    for k, v in kwargs.items():
        opts.append(f"--{k}={v}")

    spm.SentencePieceTrainer.Train(" ".join(opts))


def train_word2vec(lines,
                   spm_path,
                   **kwargs):
    """
    Train a Word2Vec model on lines with an encoded model.

    All parameters are exposed through kwargs. For a full list of parameters,
    see the gensim Word2Vec documentation:
    https://radimrehurek.com/gensim/models/word2vec.html

    Parameters
    ----------
    lines : generator
        A generator expression over lines. Usually passing open(filename) is
        sufficient.
    spm_path : str
        The path to the model file created by the spm model.

    """
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

    return reorder_embeddings(sp, model)


def train(path_to_corpus,
          spm_model_name,
          vocab_size,
          word2vec_path,
          spm_kwargs=None,
          w2v_kwargs=None):
    """
    Train an spm and a word2vec model.

    Parameters
    ----------
    path_to_corpus : str
        Path to a single text file, with one sentence/document per line.
        Any preprocessing to this file must be done beforehand, e.g, neither
        SPM nor word2vec lowercase sentences.
    spm_model_name : str
        The model name under which to save the sentencepiece model.
    vocab_size : int
        The vocab size of the spm model.
    spm_kwargs : dict
        Extra command line options to pass to the spm_train command. This takes
        the form of a dict. Command line options should be spelled exactly as
        in the official documentation. For a list of options, see:
        https://github.com/google/sentencepiece#train-sentencepiece-model
        or run `spm_train --help`
        Note that the model name, vocab size, input file and model type
        parameters are set by other flags or hardcoded.
    word2vec_path : str
        The path to which to save the word2vec model as a .vec file.
    w2v_kwargs : dict
        The kwargs to pass to the word2vec initialization function.
        The min_count parameter is always set to 0, because we want to have
        vectors for all our wordpieces.

    """
    if spm_kwargs is None:
        spm_kwargs = {}
    if w2v_kwargs is None:
        w2v_kwargs = {}
    train_spm(path_to_corpus, spm_model_name, **spm_kwargs)
    words, vectors = train_word2vec(open(path_to_corpus),
                                    f"{spm_model_name}.model",
                                    **w2v_kwargs)
    with open(word2vec_path, 'w') as f:
        shape = ' '.join([str(x) for x in vectors.shape])
        f.write(f"{shape}\n")
        for word, vec in zip(words, vectors):
            f.write(f"{word} {' '.join(str(x) for x in vec)}\n")
