import logging
from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import TweetTokenizer
from torch import nn
import torch.nn.functional as F
import os
from gensim.models import KeyedVectors

from src.utils import chunkize_iter, chunks

tokenize = TweetTokenizer().tokenize
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="4" # change GPU number depending on your machine

dims = [
    'social_support',
    'conflict',
    'trust',
    'fun',
    'similarity',
    'identity',
    'respect',
    'romance',
    'knowledge',
    'power'
]


# loads all pretrained word embeddings under the wv format using Gensim
class ExtractWordEmbeddings():
    def __init__(self, emb_type='glove',
                 emb_dir='weights/embeddings',
                 method='average'):
        from gensim.models import KeyedVectors

        emb_type = emb_type.lower()
        # if emb_type=='word2vec':
        #     load_dir = join(emb_dir,'word2vec/GoogleNews-vectors-negative300.wv')
        # elif emb_type=='fasttext':
        #     load_dir = join(emb_dir,'fasttext/wiki-news-300d-1M-subword.wv')
        if emb_type == 'glove':
            # load_dir = join(emb_dir,'glove/glove.twitter.27B.200d.wv')
            load_dir = os.path.join(emb_dir, 'glove.840B.300d.wv')

        self.model = KeyedVectors.load(load_dir, mmap='r')
        self.emb_type = emb_type
        self.method = method

        self.UNK = self.model.vectors.mean(0)  # UNK as just the average of all vectors
        if emb_type == 'word2vec':
            self.UNK = self.model['UNK']
        print("Loaded word embeddings from %s!" % load_dir)
        # print("Vocab size: %d" % len(self.model.vocab))
        return

    def fit(self, X):
        return

    # from any sentence, returns word vectors
    def obtain_vectors_from_sentence(self, words, include_unk=True):
        out = []
        for word in words:
            if word in self.model:
                vec = self.model[word]
            elif word.lower() in self.model:
                vec = self.model[word.lower()]
            else:
                if include_unk:
                    vec = self.UNK
                else:
                    continue
            out.append(vec.tolist())
        if len(out) == 0:
            return np.zeros(len(self.UNK)).reshape(1, -1)
        else:
            return np.array(out)

    def transform(self, X):
        """
        :param X: list containing
        :return:
        """
        #
        assert type(X) == list, "Error in ExtractTags: input is not a list!"
        assert type(X[0]) == list, "Error in ExtractTags: Input is not tokenized!"
        out = []
        # case 1: only 1 sentence per sample
        if type(X[0][0]) == str:
            for sentence in X:
                arr = self.obtain_vectors_from_sentence(sentence)  # sentence = list of words
                if self.method == 'average':
                    arr = np.mean(arr, axis=0)
                out.append(arr)
        # case 2: each sample has multiple sentences
        elif type(X[0][0]) == list:
            if type(X[0][0][0]) == str:
                for sentences in X:
                    all_words = []
                    for sent in sentences:
                        all_words.extend(sent)
                    arr = self.obtain_vectors_from_sentence(all_words)  # sentence = list of words
                    if self.method == 'average':
                        arr = np.mean(arr, axis=0)
                    out.append(arr)
        return out

    def get_feature_names(self):
        return ['wv-%s-%s:%d' % (self.emb_type, self.method, i) for i in range(len(self.UNK))]


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.W_out = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        """
        :param batch of size [b @ (seq x dim)]
        :return: array of size [b]
        """
        lengths = (batch != 0).sum(1)[:, 0]  # lengths of non-padded items
        lstm_outs, _ = self.lstm(batch)  # [b x seq x dim]
        out = torch.stack([lstm_outs[i, idx - 1] for i, idx in enumerate(lengths)])  # set on the last hidden state
        out = self.W_out(out)
        out = out.squeeze(-1)
        out = torch.sigmoid(out)
        return out


class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(BiLSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.lstm_f = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_b = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.W_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, batch_f, batch_b):
        """
        :param batch of size [b @ (seq x dim)]
        :return: array of size [b]
        """
        lengths = (batch_f != 0).sum(1)[:, 0]  # lengths of non-padded items
        lstm_outs, _ = self.lstm_f(batch_f)  # [b x seq x dim]
        lstm_outs2, _ = self.lstm_b(batch_b)  # [b x seq x dim]
        out = torch.stack([torch.cat([lstm_outs[i, idx - 1], lstm_outs2[i, idx - 1]]) for i, idx in enumerate(lengths)])
        out = self.W_out(out).squeeze()
        # out = torch.sigmoid(out).squeeze()
        out = torch.sigmoid(out)
        return out


def padBatch(list_of_list_of_arrays, max_seq=None):
    """
    A function that returns a numpy array that creates a B @ (seq x dim) sized-tensor
    :param list_of_list_of_arrays:
    :return:
    """

    if max_seq:
        list_of_list_of_arrays = [X[:max_seq] for X in list_of_list_of_arrays]

    # get max length
    mx = max(len(V) for V in list_of_list_of_arrays)

    # get array dimension by looking at the 1st sample
    array_dimensions = [len(V[0]) for V in list_of_list_of_arrays]
    assert min(array_dimensions) == max(array_dimensions), "Dimension sizes do not match within the samples!"
    dim_size = array_dimensions[0]

    # get empty array to put in
    dummy_arr = [0] * dim_size

    # create additional output
    out = []
    for V in list_of_list_of_arrays:
        V = V.tolist()
        out.append(V + [dummy_arr] * (mx - len(V)))

    # return
    return np.array(out)


def glove4gensim(file_dir):
    """
    A function that modifies the pretrained GloVe file so it could be integrated with this framework
    [Note] You can download the vectors used in this code at
    https://nlp.stanford.edu/projects/glove/ (make sure to unzip the files)
    :param file_dir: file directory of the downloaded file
    e.g., file_dir='/home/USERNAME/embeddings/word2vec/GoogleNews-vectors-negative300.bin'
    :return: None
    """

    from gensim.scripts.glove2word2vec import glove2word2vec
    #
    # load the vectors on gensim
    assert file_dir.endswith('.txt'), "For downloaded GloVe, the input file should be a .txt"
    # glove2word2vec(file_dir, file_dir.replace('.txt', '.vec'))
    # file_dir = file_dir.replace('.txt', '.vec')
    model = KeyedVectors.load_word2vec_format(file_dir, binary=False, no_header=True)
    # save only the .wv part of the model, it's much faster
    # new_file_dir = file_dir.replace('.vec', '.wv')
    new_file_dir = file_dir.replace('.txt', '.wv')
    model.save(new_file_dir)
    # delete the original .bin file
    # os.remove(file_dir)
    # print("Removed previous file ", file_dir)

    # try loading the new file
    model = KeyedVectors.load(new_file_dir, mmap='r')
    # print("Loaded in gensim! %d word embeddings, %d dimensions" % (len(model.vocab), len(model['a'])))
    return


def score_dimension(sentences, dim, is_cuda, model_dir, em, chunk_size=1000):
    weight_file = os.path.join(model_dir, f'LSTM/{dim}/best-weights.pth')
    # load model
    model = LSTMClassifier(embedding_dim=300, hidden_dim=300)
    state_dict = torch.load(weight_file)
    model.load_state_dict(state_dict)
    if is_cuda:
        model.cuda()

    to_return = list()
    for batch in chunks(sentences, n=chunk_size):
        vector = torch.tensor(padBatch([em.obtain_vectors_from_sentence(tokenize(sent), True) for sent in batch]),
                              device='cuda' if is_cuda else 'cpu').float()
        scores = model(vector)
        to_return.extend([i.item() for i in scores])
    return to_return


def score_dimensions(sentences, is_cuda, model_dir, chunk_size=1000):
    embedding_fpath = os.path.join(model_dir, 'glove.840B.300d.txt')
    if not os.path.exists(os.path.join(model_dir, 'glove.840B.300d.wv')):
        glove4gensim(embedding_fpath)  # change file name if using different embeddings
    # load embeddings
    em = ExtractWordEmbeddings(emb_type='glove', emb_dir=model_dir)

    scores = dict()
    for dim in dims:
        scores[dim] = score_dimension(sentences=sentences, dim=dim, is_cuda=is_cuda, model_dir=model_dir, em=em,
                                      chunk_size=chunk_size)
    to_return = [{dim: scores[dim][i] for dim in dims} for i in range(len(sentences))]
    return to_return


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    interim_dir = os.path.join(project_dir, 'data', 'interim')
    external_dir = os.path.join(project_dir, 'data', 'external')

    dim = 'conflict'
    is_cuda = True  # set to true only when using a GPU

    sents = [
        "baha'i faith makes sense once you accept it as openly hypocritical.",
        'your opinion strongly contrasts with mine',
        'i do not believe in your words',
        'i trust you',
        'i believe in you',
        'believe me, that is not going to work',
        'i love you so much',
        'i hate you',
        "I don't love you any more",
        'i am proud of you',
        'i agree with that guy',
        'Thank you so much',
        'good to hear from you',
        'this is exactly what i wanted',
        'this is not what i wanted',
        'get off, you are wrong i do not want any more of this conversation',
        'oh this is too bad'
    ]

    embedding_fpath = os.path.join(external_dir, 'glove.840B.300d.txt')
    if not os.path.exists(os.path.join(external_dir, 'glove.840B.300d.wv')):
        glove4gensim(embedding_fpath)  # change file name if using different embeddings
    # load embeddings
    em = ExtractWordEmbeddings(emb_type='glove', emb_dir=external_dir)

    scores = score_dimension(sentences=sents, dim=dim, is_cuda=is_cuda, model_dir=external_dir, em=em)
    for i in range(len(sents)):
        print(round(scores[i], 2), sents[i])

    scores = score_dimensions(sentences=sents, is_cuda=is_cuda, model_dir=external_dir)
    for sent, scores in zip(sents, scores):
        print(sent, {k: round(v, 2) for k, v in scores.items()})
