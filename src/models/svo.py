import json
import os
import re
import string

import gensim
import pandas as pd
import spacy
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from spacy.cli import download
from spacy.tokens.span import Span as SpacySpan
from spacy.parts_of_speech import CONJ, NOUN, VERB
from textacy.constants import AUX_DEPS
import numpy as np
from itertools import takewhile, islice

from textacy.spacier.utils import get_objects_of_verb, get_span_for_compound_noun, get_span_for_verb_auxiliaries, \
    get_subjects_of_verb, get_main_verbs_of_sent

N_CLUSTERS = 12

infinitify = lambda x: re.sub('[^a-zA-Z0-9]+', '_', x.lower())

pronouns = [i.lower().strip() for i in '''
all 
another 
any 
anybody 
anyone 
anything 

both 

each 
each other 
either 
everybody 
everyone 
everything 

few 

he 
her 
hers 
herself 
him 
himself 
his 

I 
it 
its 
itself 

little 

many 
me 
mine 
more 
most 
much 
my 
myself 

neither 
no one 
nobody 
none 
nothing 

one 
one another 
other 
others 
our 
ours 
ourselves 

several 
she 
some 
somebody 
someone 
something 

that 
their 
theirs 
them 
themselves 
these 
they 
this 
those 


us 

we 
what 
whatever 
which 
whichever 
who 
whoever 
whom 
whomever 
whose 

you 
your 
yours 
yourself 
yourselves '''.split() if i and len(i.strip())]
s_filter = pronouns + ['there']
v_filter = 'be been is was are were am being have has having had \'s n\'t'.split(' ')


def normalize_by_row(s_wv):
    s_wv_norms = np.linalg.norm(s_wv, axis=1)
    s_wv = (s_wv.T / s_wv_norms.reshape(1, -1)).T
    return s_wv


def remove_quotations(doc):
    quote_end_punct = {',', '.', '?', '!'}
    quote_ptrn = r'''(?:[^\w]|^)((?:".*?")|(?:''.*?'')|''' + u"(?:‘‘.*?’’)|(?:“.*?”)|(?:‘.*?’)" + '''|(?:``.*?'')|(?:`.*?')|(?:'.*?'))(?:[^\w]|$)'''
    quote_indexes = set((m.regs[-1][0], m.regs[-1][1] - 1) for m in
                        re.finditer(quote_ptrn,
                                    doc))

    quote_indexes = list(filter(lambda x:
                                not any([char in quote_end_punct for char in doc[x[0]: x[1] + 1][-4:]]),
                                # up to 2 chars of quotations, plus a blank space
                                quote_indexes))
    if (not quote_indexes) or (not len(quote_indexes)):
        return doc
    quote_indexes = sorted(quote_indexes)
    replacements = []
    for q0, q1 in quote_indexes:
        quote = doc[q0: q1 + 1]
        m = re.match(
            r'''(?:(?:"(?P<single1>.*?)")|(?:''(?P<double1>.*?)'')|''' + u"(?:‘‘(?P<double2>.*?)’’)|(?:“(?P<single5>.*?)”)|(?:‘(?P<single2>.*?)’)" + '''|(?:``(?P<double3>.*?)'')|(?:`(?P<single3>.*?)')|(?:'(?P<single4>.*?)'))''',
            quote)
        if m.lastgroup.startswith('single'):
            dequoted = quote[1:-1]
        elif m.lastgroup.startswith('double'):
            dequoted = quote[2:-2]
        replacements.append(dequoted)
    quote_indexes.insert(0, (None, -1))

    newdoc = u''''''
    for i in range(len(quote_indexes) - 1):
        newdoc += doc[quote_indexes[i][1] + 1:quote_indexes[i + 1][0]]
        newdoc += replacements[i]
    newdoc += doc[quote_indexes[-1][1] + 1:]
    return newdoc


def get_span_for_verb_xcomp(verb):
    """
    Return document indexes spanning all (adjacent) tokens
    around a verb that are auxiliary verbs or negations.
    """
    AUX_XCOMP_DEPS = AUX_DEPS.copy()
    AUX_XCOMP_DEPS.add(u'xcomp')
    min_i = verb.i - sum(1 for _ in takewhile(lambda x: x.dep_ in AUX_XCOMP_DEPS,
                                              reversed(list(verb.lefts))))

    max_i = verb.i

    for k in verb.subtree:
        if k.dep_ in AUX_XCOMP_DEPS and (k.i == max_i + 1):  # only adjacent
            max_i += 1

    return (min_i, max_i)


def subject_verb_object_triples(doc):
    """
    Extract an ordered sequence of subject-verb-object (SVO) triples from a
    spacy-parsed doc. Note that this only works for SVO languages.

    Args:
        doc (``textacy.Doc`` or ``spacy.Doc`` or ``spacy.Span``)

    Yields:
        Tuple[``spacy.Span``, ``spacy.Span``, ``spacy.Span``]: the next 3-tuple
            of spans from ``doc`` representing a (subject, verb, object) triple,
            in order of appearance
    """
    # TODO: What to do about questions, where it may be VSO instead of SVO?
    # TODO: What about non-adjacent verb negations?
    # TODO: What about object (noun) negations?
    if isinstance(doc, SpacySpan):
        sents = [doc]
    else:  # textacy.Doc or spacy.Doc
        sents = doc.sents

    for sent in sents:
        start_i = sent[0].i

        verbs = get_main_verbs_of_sent(sent)
        for verb in verbs:
            subjs = get_subjects_of_verb(verb)
            if not subjs:
                continue
            objs = get_objects_of_verb(verb)
            if not objs:
                continue

            # add adjacent auxiliaries to verbs, for context
            # and add compounds to compound nouns
            verb_span = get_span_for_verb_xcomp(verb)
            if verb_span[0] != verb_span[1]:
                # we've got a runner: xcomp verb. update obj
                #                 print sent[verb_span[0] - start_i: verb_span[1] - start_i + 1]
                newverb = sorted([i
                                  for i in sent[verb_span[0] - start_i: verb_span[1] - start_i + 1]
                                  if i.pos == spacy.parts_of_speech.VERB], key=lambda i: i.i)
                newverb = newverb[-1]
                objs = get_objects_of_verb(newverb)
                if not objs:
                    continue

            verb = sent[verb_span[0] - start_i: verb_span[1] - start_i + 1]
            for subj in subjs:
                subj = sent[get_span_for_compound_noun(subj)[0] - start_i: subj.i - start_i + 1]
                for obj in objs:
                    if obj.pos == NOUN:
                        span = get_span_for_compound_noun(obj)
                    elif obj.pos == VERB:
                        span = get_span_for_verb_auxiliaries(obj)
                    else:
                        span = (obj.i, obj.i)
                    obj = sent[span[0] - start_i: span[1] - start_i + 1]

                    yield (subj, verb, obj)


def generate_triples_with_lines(sentences, parser):
    for sentence in sentences:
        if sentence and sentence.strip():
            for triple in subject_verb_object_triples(parser(sentence)):
                yield triple, sentence


def infinity_triple_in_wv(svo, wv):
    (s, v, o) = svo
    return (s in wv.wv) and (v in wv.wv) and (o in wv.wv)


def build_concatenated_v_vector(unique_triples_infty_wv):
    s_wv = np.array([np.concatenate(i) for i in unique_triples_infty_wv])
    s_wv[:, 100:200] -= s_wv[:, :100]
    s_wv[:, 200:] -= s_wv[:, :100]
    s_wv = np.hstack((normalize_by_row(s_wv[:, :100]),
                      normalize_by_row(s_wv[:, 100:200]),
                      normalize_by_row(s_wv[:, 200:])
                      ))

    return s_wv


def triple_entity_is_pronoun(svo, entity_idx):
    entity = svo[entity_idx]
    return len(entity) == 1 and entity[0].pos == spacy.parts_of_speech.PRON


def to_text(entity):
    no_heading = re.sub(r'^\W+', '', entity.lemma_)
    no_trailing_either = re.sub(r'\W+$', '', no_heading)
    return infinitify(no_trailing_either)


def svo_in_wv(l, svo, wv):
    return infinity_triple_in_wv(svo, wv)


def svo_to_text(l, svo):
    return (l, (to_text(svo[0]), to_text(svo[1]), to_text(svo[2])))


def o_end_not_verb(svo):
    return svo[2][-1].pos != spacy.parts_of_speech.VERB


def s_not_pronoun(svo):
    return not triple_entity_is_pronoun(svo, 0)


def o_not_pronoun(svo):
    return not triple_entity_is_pronoun(svo, 2)


def s_not_generic(svo):
    return svo[0].text.lower() not in s_filter


##### for infinitygrams
from nltk.tag import PerceptronTagger
import ftfy
from nltk.corpus import stopwords
from nltk import TweetTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize
from gensim.models.word2vec import Word2Vec, LineSentence


def analyze(newline, pattern):
    return re.findall(pattern, newline)


def construct_ngrams(analyzed):
    return [re.sub(r'[^a-zA-Z0-9]', '_', x) for x in analyzed]


def format_lines(lines, pattern):
    for line in lines:
        yield ' '.join(construct_ngrams(analyze(line, pattern)))  # check this returns token in correct order


remove_urls = lambda x: re.sub(r"http\S+", "", x)


def stream_docs(fpath):
    with open(fpath, encoding='utf8') as f:
        for i in f:
            yield i


def doc_to_json(doc):
    return json.loads(doc)


def get_field(doc, field):
    return doc[field]


stop_words_ = stopwords.words('english') + list(string.punctuation)


def stream_field(fpath, field):
    for doc in stream_docs(fpath):
        yield get_field(doc_to_json(doc), field)


def preproc(doc):
    return ftfy.fix_text(remove_urls(doc))


def penn2morphy(penntag, returnNone=True):
    morphy_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
                  'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''


class LemmaTokenizer(object):
    def __init__(self, case_sensitive=False, stop_words=None):
        self.pct = PerceptronTagger()
        self.stm = PorterStemmer()
        self.wnl = WordNetLemmatizer()
        self.tkn = TweetTokenizer(preserve_case=case_sensitive)
        self.case_Sensitive = case_sensitive
        self.stop_words = stop_words and stop_words or []

    def stem(self, t):
        if t.startswith('#'):  # preserve hashtags
            return t
        else:
            stemmed = self.stm.stem(t)
            if len(stemmed) > 2:
                return stemmed
            else:
                return t

    def lemmatize(self, tokenized):
        for word, pos in self.pct.tag(tokenized):
            pos_converted = penn2morphy(pos)
            if pos_converted:
                yield self.wnl.lemmatize(word, pos_converted)
            else:
                yield self.wnl.lemmatize(word)

    def __call__(self, doc):
        tokenized = self.tkn.tokenize(preproc(doc).lower())
        lemmatized = self.lemmatize(tokenized)
        return [t for t in lemmatized if (len(t) > 1)]


def stream_all_docs(fpath):
    for base, dirs, files in os.walk(fpath):
        if files:
            for f in files:
                for d in stream_docs(os.path.join(base, f)):
                    yield d


def stream_all_docs_field(fpath, field):
    for d in stream_all_docs(fpath):
        yield json.loads(d)[field]


def create_w2v_sentences_infinity(texts, pattern, outfpath):
    with open(outfpath, 'w+', encoding='utf8') as f:
        for sent in format_lines(texts, pattern):
            f.write(sent + '\n')


def stream_sentences(txt):
    for sent in sent_tokenize(txt):
        yield sent


def lemmatize(inpath, field):
    lemma = LemmaTokenizer(case_sensitive=False, stop_words=stop_words_)
    for i in stream_all_docs_field(inpath, field):
        for s in stream_sentences(i):
            yield ' '.join(lemma(s))


if __name__ == '__main__':
    with open('titles.txt', encoding='utf8') as f:
        lines = list(islice(f, 100))
    try:
        parser = spacy.load('en_core_web_lg')
    except OSError:
        download(model="en_core_web_lg")
        parser = spacy.load('en_core_web_lg')

    # # run the following lines to clean the input to infinitygrams:
    # filtered_lines = set(filter(lambda x: re.match(r'^ .* \t', re.sub(' (0 )+', '', x)), lines))
    # filtered_lines = filter(lambda x: len(x) > 1, set(i.split('\t')[0].strip() for i in filtered_lines))
    # filtered_lines = filter(lambda x: not all([(i in stop_words_) for i in x.split(' ')]), filtered_lines)
    #
    # pattern = re.compile(r'\b(' + r'|'.join(
    #     sorted([re.escape(r'%s' % i) for i in filtered_lines], key=lambda x: len(x), reverse=True)) + r')\b', re.DOTALL)
    # inputf = 'w2v_comments_infinity_withstops.txt'
    # create_w2v_sentences_infinity(lemmatize('../scrape_reddit/conspiracy/', 'body'),
    #                               pattern,
    #                               inputf)
    # # I extracted infinitygrams with an external cpp-based tool, which you can find here: https://github.com/lkevers/ldig-python3
    # # run the following lines to train the w2v model on infinitygrams
    # sentences_infty = LineSentence(inputf)
    # w2v_infty = Word2Vec(sentences_infty, workers=78)
    # w2v_infty.save('w2v_infinity_withstops_submissions')

    print('\n'.join(lines[:3]))

    triples, corresponding_lines = zip(
        *list(generate_triples_with_lines(set(remove_quotations(l).strip() for l in lines), parser)))
    print(triples, corresponding_lines)
    corresponding_lines_nopron, triples_nopron = zip(
        *filter(lambda x: s_not_pronoun(x[1]), zip(corresponding_lines, triples)))
    corresponding_lines_nopron, triples_nopron = zip(
        *filter(lambda x: o_not_pronoun(x[1]), zip(corresponding_lines_nopron, triples_nopron)))
    corresponding_lines_noverbo, triples_noverbo = zip(
        *filter(lambda x: o_end_not_verb(x[1]), zip(corresponding_lines_nopron, triples_nopron)))
    corresponding_lines_nogeneric, triples_nogeneric = zip(
        *filter(lambda x: s_not_generic(x[1]), zip(corresponding_lines_noverbo, triples_noverbo)))
    corresponding_lines_text, triples_text = zip(
        *map(lambda x: svo_to_text(x[0], x[1]), zip(corresponding_lines_nogeneric, triples_nogeneric)))

    for l, t in islice(zip(corresponding_lines_text, triples_text), 10):
        print(l, t)
    wv = gensim.models.Word2Vec.load('w2v_infinity_withstops_comments')
    corresponding_lines_inwv, triples_inwv = zip(
        *filter(lambda x: svo_in_wv(x[0], x[1], wv=wv), zip(corresponding_lines_text, triples_text)))
    triples_infty_wv = [(wv.wv[s], wv.wv[v], wv.wv[o]) for (s, v, o) in triples_inwv]

    s_wv_v = build_concatenated_v_vector(triples_infty_wv)
    X_v = SimpleImputer().fit_transform(s_wv_v)

    model = KMeans(n_clusters=N_CLUSTERS, n_init='auto').fit(X_v)
    prediction = model.predict(X_v)
    cluster_centers = dict(enumerate(model.cluster_centers_))

    df = pd.DataFrame({'cluster': prediction,
                       'subj': [i[0] for i in triples_inwv],
                       'verb': [i[1] for i in triples_inwv],
                       'obj': [i[2] for i in triples_inwv]})
    print(df)
