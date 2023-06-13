import os
from collections import OrderedDict
import time

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE, Isomap

from src.features.gensim_word2vec_procrustes_align import load_embeddings

CMAP_MIN = 5


# adapted from https://github.com/jacobeisenstein/language-change-tutorial/blob/master/ic2s2-notebooks/DirtyLaundering.ipynb
# and https://github.com/williamleif/histwords/blob/master/viz/scripts/closest_over_time_with_anns.py
def get_cmap(n, name='YlGn'):
    matplotlib.colormaps[name]
    return plt.cm.get_cmap(name, n + CMAP_MIN)


def fit_tsne(values):
    if not values:
        return

    start = time.time()
    mat = np.array(values)
    model = TSNE(n_components=2, random_state=0, learning_rate=150, init='random')
    # model = TSNE(n_components=2, random_state=0, learning_rate=150, init='pca')
    # model = Isomap(n_components=2)
    # fitted = model.fit(mat).transform(to_transform)

    fitted = model.fit_transform(mat)
    print("FIT TSNE TOOK %s" % (time.time() - start))

    return fitted


def get_time_sims(embed_dict, word1, min_sim=0.3, n_neighbors=15,
                  blacklisted_words={'conspiracy', 'theory', 'conspiracist', 'conspiracy_theorist', 'theorist'}):
    start = time.time()
    time_sims = OrderedDict()
    lookups = {}
    nearests = {}
    sims = {}
    for year, embed in embed_dict.items():
        nearest = []
        nearests["%s|%s" % (word1, year)] = nearest
        time_sims[year] = []
        if word1 not in embed.wv.key_to_index:
            print(f"{word1} not in {year}")
            continue
        cnt = 0
        for (word, sim) in [(word1, 1.)] + embed.wv.similar_by_word(word1, topn=n_neighbors + len(blacklisted_words)):
            if word in blacklisted_words:
                continue
            cnt += 1
            if cnt > n_neighbors: break
            ww = "%s|%s" % (word, year)
            nearest.append((sim, ww))
            if sim > min_sim:
                time_sims[year].append((sim, ww))
                lookups[ww] = embed.wv[word]
                sims[ww] = sim

    print("GET TIME SIMS FOR %s TOOK %s" % (word1, time.time() - start))
    return time_sims, lookups, nearests, sims


def plot_historical_neighbors(word1, min_sim=.3, n_neighbors=15,
                              embedding_dir='../../data/interim/aligned_embeddings/sample_and_labeling',
                              figure_dir='../../reports/figures',
                              blacklisted_words={'conspiracy', 'theory', 'conspiracist', 'conspiracy_theorist',
                                                 'theorist'}):
    embeddings = load_embeddings(embedding_dir)
    time_sims, lookups, nearests, sims = get_time_sims(embeddings, word1, min_sim, n_neighbors,
                                                       blacklisted_words=blacklisted_words)
    dirname = os.path.split(embedding_dir)[-1]

    words = lookups.keys()
    # values = [lookups[word] for word in words if not word.startswith(word1)]
    to_transform = [lookups[word] for word in words]
    fitted = fit_tsne(to_transform)
    if (fitted is None) or (not len(fitted)):
        print("Couldn't model word", word1)
        return

    # draw the words onto the graph
    cmap = get_cmap(len(time_sims))
    annotations = plot_words(word1, words, fitted, cmap, sims)

    if annotations:
        plot_annotations(annotations)

    savefig(f"{word1}_annotated_{dirname}", figure_dir)
    for year, sim in time_sims.items():
        print(year, sim)


def plot_words(word1, words, fitted, cmap, sims):
    # TODO: remove this and just set the plot axes directly
    plt.cla()
    plt.clf()
    plt.scatter(fitted[:, 0], fitted[:, 1], alpha=0)
    plt.suptitle("%s" % word1, fontsize=30, y=0.1)
    plt.axis('off')
    words = list(words)

    annotations = []
    isArray = type(word1) == list
    for i in range(len(words)):
        pt = fitted[i]

        ww, year = [w.strip() for w in words[i].split("|")]
        color = cmap((int(year) - 2012) + CMAP_MIN)
        word = ww
        sizing = sims[words[i]] * 20

        # word1 is the word we are plotting against
        if ww == word1 or (isArray and ww in word1):
            annotations.append((ww, year, pt))
            word = year
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
        # print(prev, ann)
        prev = ann


def savefig(name, directory='../../reports/figures'):
    fname = os.path.join(directory, name)
    plt.savefig(fname, bbox_inches=0)


if __name__ == '__main__':
    aligned_embedding_dir = '../../data/interim/aligned_embeddings'

    for dirname in os.listdir(aligned_embedding_dir):
        for term in {'conspiracist', 'conspiracy_theorist', 'conspiracy'}:
            print(dirname, term)
            blacklisted_words = {'conspiracy', 'theory', 'conspiracist', 'conspiracy_theorist', 'theorist'}
            blacklisted_words.remove(term)
            plot_historical_neighbors(term,
                                      n_neighbors=5,
                                      embedding_dir=os.path.join(aligned_embedding_dir, dirname),
                                      blacklisted_words=blacklisted_words)
        for term in {'trump', 'sanders'}:
            print(dirname, term)
            plot_historical_neighbors(term,
                                      n_neighbors=10,
                                      embedding_dir=os.path.join(aligned_embedding_dir, dirname),
                                      blacklisted_words={})
