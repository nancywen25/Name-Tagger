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
    Given a sentence,
    generate a feature vector for each token

    The features may involve previous, current and next tokens,
    parts-of-speech, and chunk tags, as well as combinations of these
    Args:
        sentence:

    Returns:

    """

    basic_features = []

    for token in sentence:
        d = {"token": token[0],
            "pos": token[1],
            "chunk": token[2],
            "tag": None}

        if len(token) == 4: # includes the name tag
            d["tag"] = token[3]

        basic_features.append(d)

    return basic_features


def enhance_features():
    """
    Read in a .pos-chunk or .pos-chunk-name file,
    and augment with additional features

    Returns:

    """
    feature_list = []
    for sentence in sentences:
        basic_features = feature_builder(sentence)
        feature_list += basic_features

    feature_df = pd.DataFrame(feature_list)


fname_train = "data/CONLL_train.pos-chunk-name"
sentences = read_data(fname_train)

