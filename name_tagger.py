import pandas as pd
import itertools

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
            "chunk": token[2]}

        if len(token) == 4: # includes the name tag
            d["tag"] = token[3]

        basic_features.append(d)

    return basic_features


def extract_features(sentences):
    """
    Read in a .pos-chunk or .pos-chunk-name file,
    and augment with additional features

    Returns:

    """
    feature_list = []
    for sentence in sentences:
        basic_features = feature_builder(sentence)
        feature_list += basic_features

    return pd.DataFrame(feature_list)

def get_tags(sentences):
    """
    Given data with just token and tags,
    returns the data as a dataframe with token and tag columns

    Args:
        sentences:

    Returns:

    """
    tag_list = []
    for sentence in sentences:
        for token in sentence:
            tag_list.append({'token': token[0],
                             'tag': token[1]})
    return pd.DataFrame(tag_list)

def me_train(df):
    """
    Given a dataframe representing features and tag values,
    trains a MaxEnt model to predict tag value

    Args:
        df: dataframe containing features and tag values

    Returns:
        lr: trained LogisticRegression moel
        vec: trained feature vectorizer
    """
    # encode the features
    vec = DictVectorizer()
    features = ["token", "pos", "chunk"]
    X_train = vec.fit_transform(df[features].to_dict("records")) # change df back to list of dicts
    # vec.get_feature_names()

    # encode the tag categories
    y_train = df["tag"].values

    # train the model
    logreg = LogisticRegression(multi_class="multinomial",
                                max_iter=1000)
    lr = logreg.fit(X_train, y_train)
    print("Accuracy:", lr.score(X_train, y_train)) # get in-sample accuracy

    return lr, vec

def me_tag(lr, vec, df):
    features = ["token", "pos", "chunk"]
    X = vec.transform(df[features].to_dict("records"))
    y_pred = lr.predict(X)

    return y_pred

def write_data(fname_in, fname_out, df, y):
    """
    Write the predicted tags out to a .name file
    Returns:

    """
    tokens = df['token'].values
    i = 0
    with open(fname_in, 'r') as infile:
        with open(fname_out, 'w') as outfile:
            for line in infile:
                if line == '\n': # empty line
                    outfile.write('\n')
                elif i == len(tokens):
                    continue
                else:
                    outfile.write(tokens[i] + "\t" + y[i] + "\n")
                    i += 1

# TRAIN: training model using train data
fname_train = "data/CONLL_train.pos-chunk-name"
sentences = read_data(fname_train)
df = extract_features(sentences)
lr, vec = me_train(df)

# DEV: use model to label dev data
fname_dev = "data/CONLL_dev.pos-chunk"
fname_dev_tag = "data/CONLL_dev.name"
sentences_dev = read_data(fname_dev)
df_dev = extract_features(sentences_dev)

tag_dev = read_data(fname_dev_tag)
df_tags = get_tags(tag_dev)

# write predictions to file
y_dev = me_tag(lr, vec, df_dev)
write_data("data/CONLL_dev.pos-chunk", "output/CONLL_dev.name", df_dev, y_dev)

# evaluating dev performance
y_true = df_dev["tag"].values
print(accuracy_score(y_true, y_dev))

# TEST: generate output for test data
fname_test = "data/CONLL_test.pos-chunk"
sentences_test = read_data(fname_test)
df_test = extract_features(sentences_test)
y_test = me_tag(lr, vec, df_test)
write_data("data/CONLL_test.pos-chunk", "output/CONLL_test.name", df_test, y_test)




