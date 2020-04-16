import pandas as pd
import itertools

def read_data(fname):
    """
    Given a file name of .pos-chunk or .pos-chunk-name format,
    Returns a list of sentences in that file

    Args:
        fname:

    Returns:
        list of list of list
    """
    lines = []
    # each element in a line can be token or empty line
    with open(fname, 'r') as f:
        for line in f:
            lines.append(line.split())

    # split list into list of sentences
    sentences = [list(v) for k, v in itertools.groupby(lines, key=bool) if k]

    # TODO: should we can drop the -DOCSTART- "sentences"?
    return sentences


def feature_builder(sentence):
    """
    Given a sentence, generate features for each token in a sentence

    The features may involve previous, current and next tokens,
    parts-of-speech, and chunk tags, as well as combinations of these
    Args:
        token:

    Returns:

    """
    pass

def enhance_features():
    """
    Read in a .pos-chunk or .pos-chunk-name file,
    and augment with additional features

    Returns:

    """
    pass

