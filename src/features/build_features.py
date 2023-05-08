import datetime
import json
import logging
import os
import pickle
import re
from functools import partial
from itertools import islice
from multiprocessing.pool import Pool
from pathlib import Path

from gensim.models import Word2Vec
from gensim.test.utils import datapath

from src.data.make_dataset import CONSPIRACY_THEORIST_RE
from src.features.preprocess_text import clean_items, preprocess_pre_tokenizing
from src.utils import to_file

from langdetect import detect


def normalize_text(item: dict, **kwargs):
    contribution_type = "comment"
    if 'is_self' in item:
        if item['is_self']:
            contribution_type = "selftext_submission"
        else:
            contribution_type = "link_submission"
    item['contribution_type'] = contribution_type

    if contribution_type == 'comment':
        item['text'] = item.pop('body')
    elif contribution_type == 'selftext_submission':
        item['text'] = "\n".join([item.pop('title'), item.pop('selftext')])
    elif contribution_type == 'link_submission':
        item['text'] = item.pop('title')
    # title if link sub; title+self if self sub; text if comment

    if contribution_type in {'selftext_submission', 'link_submission'}:
        item['fullname'] = "t3_" + item.pop('id')
        item['parent_fullname'] = None
        item['link_fullname'] = item['fullname']
    elif contribution_type == 'comment':
        item['fullname'] = "t1_" + item.pop('id')
        item['parent_fullname'] = item.pop('parent_id')
        item['link_fullname'] = item.pop('link_id')
    return item


def preprocess(item, text_field='text', preprocessed_field='preprocessed_text'):
    text = item[text_field]
    # text = re.sub(r'conspiracy.theorists?', 'conspiracy_theorist', text, flags=re.I|re.U|re.DOTALL|re.M)
    item[preprocessed_field] = preprocess_pre_tokenizing(text)
    return item


def stream_normalized_contribution(fpath):
    with open(fpath, encoding='utf8') as f:
        yield from map(normalize_text, map(json.loads, f))


def detect_language(item, language='en', text_field='text'):
    try:
        return (detect(item[text_field]) == language, item)
    except:
        return (False, item)


def filter_language(item_stream, language='en', text_field='text',
                    n_processors=40):
    with Pool(n_processors) as pool:
        for keep, item in pool.imap(
                partial(detect_language, language=language,
                        text_field=text_field),
                item_stream):
            if keep:
                yield item


def detect_discussions(item, filter_values, filter_field):
    return (item[filter_field] in filter_values, item)


def filter_discussions(item_stream, discussions,
                       n_processors=40, filter_field='link_fullname'):
    with Pool(n_processors) as pool:
        for keep, item in pool.imap_unordered(
                partial(detect_discussions, filter_values=discussions,
                        filter_field=filter_field),
                item_stream):
            if keep:
                yield item


def preprocess_files():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]

    interim_dir = os.path.join(project_dir, 'data', 'interim')
    labeling_fpath = os.path.join(interim_dir,
                                  'labeling_contributions.jsonl')
    discussion_fpath = os.path.join(interim_dir,
                                    'labeling_discussions_all.jsonl')
    k = 100000
    sample_fpath = os.path.join(interim_dir,
                                f'sample_contributions_{k}.jsonl')
    ct_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                   f'sample_contributions_{k}_ct.jsonl')
    default_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                        f'sample_contributions_{k}_default.jsonl')

    # fpath = labeling_fpath
    # out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    # to_file(out_fpath, clean_items(item_stream=
    #                                filter(lambda item: re.findall(
    #                                    CONSPIRACY_THEORIST_RE,
    #                                    item['preprocessed_text'],
    #                                    flags=re.I | re.DOTALL | re.U | re.M),
    #                                       map(preprocess,
    #                                           stream_normalized_contribution(
    #                                               fpath))),
    #                                text_field='preprocessed_text',
    #                                cleaned_text_field='processed_text',
    #                                remove_punct=True, remove_digit=True,
    #                                remove_stops=True,
    #                                remove_pron=False,
    #                                lemmatize=True, lowercase=True,
    #                                n_process=-1
    #                                ))

    # fpath = sample_fpath
    # out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    # # keep only English contributions in the random sample
    # with Pool(40) as pool:
    #     to_file(out_fpath, clean_items(item_stream=filter_language(
    #           pool.imap_unordered(preprocess,
    #                               stream_normalized_contribution(
    #                                   fpath))),
    #                                    text_field='preprocessed_text',
    #                                    cleaned_text_field='processed_text',
    #                                    remove_punct=True, remove_digit=True,
    #                                    remove_stops=True,
    #                                    remove_pron=False,
    #                                    lemmatize=True, lowercase=True,
    #                                    n_process=-1
    #                                    ))

    fpath = ct_sample_fpath
    out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    with Pool(40) as pool:
        to_file(out_fpath, clean_items(item_stream=
                                       pool.imap_unordered(preprocess,
                                                           stream_normalized_contribution(
                                                               fpath)),
                                       text_field='preprocessed_text',
                                       cleaned_text_field='processed_text',
                                       remove_punct=True, remove_digit=True,
                                       remove_stops=True,
                                       remove_pron=False,
                                       lemmatize=True, lowercase=True,
                                       n_process=-1
                                       ))

    fpath = default_sample_fpath
    out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    with Pool(40) as pool:
        to_file(out_fpath, clean_items(item_stream=
                                       pool.imap_unordered(preprocess,
                                                           stream_normalized_contribution(
                                                               fpath)),
                                       text_field='preprocessed_text',
                                       cleaned_text_field='processed_text',
                                       remove_punct=True, remove_digit=True,
                                       remove_stops=True,
                                       remove_pron=False,
                                       lemmatize=True, lowercase=True,
                                       n_process=-1
                                       ))

    # read discussions filtered after preprocessing
    discussion_id_fpath = os.path.join(interim_dir, 'discussion_ids.pkl')
    if os.path.exists(discussion_id_fpath):
        with open(discussion_id_fpath, 'rb') as f:
            discussions = pickle.load(f)
    else:
        with open(out_fpath, encoding='utf8') as f:
            discussions = set(
                i['link_fullname'] for i in
                # i['name'] if ('selftext' in i) else i['link_id'] for i in
                map(json.loads, f))
        with open(discussion_id_fpath, 'wb+') as f:
            pickle.dump(discussions, f)
    print(f'loaded {len(discussions)} discussion ids')
    fpath = discussion_fpath
    out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    # preprocess only filtered discussions
    with Pool(40) as pool:
        to_file(out_fpath, clean_items(item_stream=
                                       pool.imap_unordered(preprocess,
                                                           filter_discussions(
                                                               stream_normalized_contribution(
                                                                   fpath),
                                                               discussions)),
                                       text_field='preprocessed_text',
                                       cleaned_text_field='processed_text',
                                       remove_punct=True, remove_digit=True,
                                       remove_stops=True,
                                       remove_pron=False,
                                       lemmatize=True, lowercase=True,
                                       n_process=-1
                                       ))


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, fpath):
        self.fpath = fpath

    def __iter__(self):
        corpus_path = datapath(self.fpath)
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield line.split()


def build_embeddings():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()
    # prepare input for the embeddings
    project_dir = Path(__file__).resolve().parents[2]

    interim_dir = os.path.join(project_dir, 'data', 'interim')
    labeling_fpath = os.path.join(interim_dir,
                                  'labeling_contributions_preprocessed.jsonl')
    k = 100000
    sample_fpath = os.path.join(interim_dir,
                                f'sample_contributions_{k}_preprocessed.jsonl')
    ct_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                   f'sample_contributions_{k}_ct_preprocessed.jsonl')
    default_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                        f'sample_contributions_{k}_default_preprocessed.jsonl')
    out_fhandles = dict()
    os.makedirs(os.path.join(interim_dir, 'text_years'), exist_ok=True)
    for folder_name, input_fpath in [("labeling", labeling_fpath),
                                     ("sample", sample_fpath),
                                     ("ct_sample", ct_sample_fpath),
                                     ("default_sample", default_sample_fpath)]:
        with open(input_fpath, encoding='utf8') as f:
            os.makedirs(os.path.join(interim_dir, 'text_years', folder_name),
                        exist_ok=True)
            logger.info(
                f"preparing {os.path.join(interim_dir, 'text_years', folder_name)}")
            for item in map(json.loads, f):
                item_date = datetime.datetime.fromtimestamp(
                    float(item['created_utc']))
                item_year = item_date.year
                item_text = item['processed_text']
                item_text = re.sub(r'conspiracy.theorists?',
                                   'conspiracy_theorist', item_text,
                                   flags=re.I | re.U | re.DOTALL | re.M)
                if item_year not in out_fhandles:
                    year_path = os.path.join(interim_dir, 'text_years',
                                             folder_name,
                                             f'{item_year}.csv')
                    out_fhandles[item_year] = open(year_path
                                                   , "w+", encoding='utf8')
                    logger.info(f"opening {year_path}")
                ff = out_fhandles[item_year]
                ff.write(item_text + '\n')
    for ff in out_fhandles.values():
        ff.close()

    # train embeddings
    os.makedirs(os.path.join(interim_dir, 'embeddings'), exist_ok=True)
    for dirname in os.listdir(os.path.join(interim_dir, 'text_years')):
        for fname in os.listdir(
                os.path.join(interim_dir, 'text_years', dirname)):
            fpath = os.path.join(interim_dir, 'text_years', dirname, fname)
            logger.info(f"training vectors for {fpath}")
            year = int(fpath[-len('.csv') - 4:-len('.csv')])
            corpus = MyCorpus(fpath)
            model = Word2Vec(sentences=corpus, seed=42, epochs=10)
            os.makedirs(os.path.join(interim_dir, 'embeddings', dirname),
                        exist_ok=True)

            model.save(
                os.path.join(interim_dir, 'embeddings', dirname,
                             f"word2vec_{year}.model"))
            word_vectors = model.wv
            word_vectors.save(os.path.join(interim_dir, 'embeddings', dirname,
                                           f"word2vec_{year}.wordvectors"))


if __name__ == '__main__':
    # preprocess_files()
    build_embeddings()
