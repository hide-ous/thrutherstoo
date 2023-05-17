import os
from collections import OrderedDict
import time

import numpy as np
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

CMAP_MIN = 5


# adapted from https://github.com/jacobeisenstein/language-change-tutorial/blob/master/ic2s2-notebooks/DirtyLaundering.ipynb
# and https://github.com/williamleif/histwords/blob/master/viz/scripts/closest_over_time_with_anns.py
def get_cmap(n, name='YlGn'):
    return plt.cm.get_cmap(name, n + CMAP_MIN)


def load_embeddings(dirpath='../../models/embeddings/'):
    to_return = OrderedDict()
    for year_str in os.listdir(dirpath):
        year = int(year_str)
        to_return[year] = Word2Vec.load(
            os.path.join(dirpath,
                         f"word2vec_{year}.model"))
    return to_return


def fit_tsne(values):
    if not values:
        return

    start = time.time()
    mat = np.array(values)
    model = TSNE(n_components=2, random_state=0, learning_rate=150, init='pca')
    fitted = model.fit_transform(mat)
    print("FIT TSNE TOOK %s" % (time.time() - start))

    return fitted


def get_time_sims(embed_dict, word1, min_sim=0.3):
    start = time.time()
    time_sims = OrderedDict()
    lookups = {}
    nearests = {}
    sims = {}
    for year, embed in embed_dict.iteritems():
        nearest = []
        nearests["%s|%s" % (word1, year)] = nearest
        time_sims[year] = []

        for (word, sim) in embed.wv.similar_by_word(word1, topn=15):
            ww = "%s|%s" % (word, year)
            nearest.append((sim, ww))
            if sim > min_sim:
                time_sims[year].append((sim, ww))
                lookups[ww] = embed.wv[word]
                sims[ww] = sim

    print("GET TIME SIMS FOR %s TOOK %s" % (word1, time.time() - start))
    return time_sims, lookups, nearests, sims


def plot_hist(word1):
    embeddings = load_embeddings()
    time_sims, lookups, nearests, sims = get_time_sims(embeddings, word1)

    words = lookups.keys()
    values = [lookups[word] for word in words]
    fitted = fit_tsne(values)
    if not len(fitted):
        print("Couldn't model word", word1)
        return

    # draw the words onto the graph
    cmap = get_cmap(len(time_sims))
    annotations = plot_words(word1, words, fitted, cmap, sims)

    if annotations:
        plot_annotations(annotations)

    savefig("%s_annotated" % word1)
    for year, sim in time_sims.iteritems():
        print(year, sim)


def plot_words(word1, words, fitted, cmap, sims):
    # TODO: remove this and just set the plot axes directly
    plt.scatter(fitted[:, 0], fitted[:, 1], alpha=0)
    plt.suptitle("%s" % word1, fontsize=30, y=0.1)
    plt.axis('off')

    annotations = []
    isArray = type(word1) == list
    for i in range(len(words)):
        pt = fitted[i]

        ww, decade = [w.strip() for w in words[i].split("|")]
        color = cmap((int(decade) - 1840) / 10 + CMAP_MIN)
        word = ww
        sizing = sims[words[i]] * 30

        # word1 is the word we are plotting against
        if ww == word1 or (isArray and ww in word1):
            annotations.append((ww, decade, pt))
            word = decade
            color = 'black'
            sizing = 15

        plt.text(pt[0], pt[1], word, color=color, size=int(sizing))

    return annotations


def plot_annotations(annotations):
    # draw the movement between the word through the decades as a series of
    # annotations on the graph
    annotations.sort(key=lambda w: w[1], reverse=True)
    prev = annotations[0][-1]
    for ww, decade, ann in annotations[1:]:
        plt.annotate('', xy=prev, xytext=ann,
                     arrowprops=dict(facecolor='blue', shrink=0.1, alpha=0.3,
                                     width=2, headwidth=15))
        print(prev, ann)
        prev = ann


def savefig(name):
    directory = '../../reports/figures'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = os.path.join(directory, name)

    plt.savefig(fname, bbox_inches=0)


if __name__ == '__main__':
    plot_hist('conspiracy_theorist')
