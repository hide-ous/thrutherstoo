import datetime
import json
import logging
import os
import pickle
import random
import re
import time
from functools import partial
from itertools import islice
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath
from googleapiclient import discovery
from tqdm import tqdm

from src.data.make_dataset import CONSPIRACY_THEORIST_RE, CONSPIRACY_SUBREDDITS, \
    DEFAULT_SUBREDDITS
from src.features.gensim_word2vec_procrustes_align import align_years
from src.features.liwcifer import read_liwc, df_liwcifer, \
    get_matchers
from src.features.perspective import get_toxicity_score, \
    REQUESTED_ATTRIBUTES_TOXICITY, REQUESTED_ATTRIBUTES_ALL
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


def parse_and_normalize(item_str):
    return normalize_text(json.loads(item_str))


def stream_normalized_contribution(fpath):
    # with open(fpath, encoding='utf8') as f, Pool(40) as pool:
    #     yield from pool.imap_unordered(parse_and_normalize, f)
    with open(fpath, encoding='utf8') as f:
        yield from map(parse_and_normalize, f)


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


def detect_multiple_discussions(items, filter_values, filter_field):
    return [item for item in items if item[filter_field] in filter_values]


def chunkize_iter(in_stream, chunk_size=1000):
    while len(chunk := list(islice(in_stream, chunk_size))):
        yield chunk


def filter_discussions(item_stream, discussions,
                       n_processors=40, filter_field='link_fullname',
                       chunk_size=10000):
    # with Pool(n_processors) as pool:
    #     for items in pool.imap_unordered(
    #             partial(detect_multiple_discussions, filter_values=discussions,
    #                     filter_field=filter_field),
    #             chunkize_iter(item_stream, chunk_size=chunk_size)):
    #         yield from items

    # with Pool(n_processors) as pool:
    #     for keep, item in pool.imap_unordered(
    #             partial(detect_discussions, filter_values=discussions,
    #                     filter_field=filter_field),
    #             item_stream):
    #         if keep:
    #             yield item

    for keep, item in map(
            partial(detect_discussions, filter_values=discussions,
                    filter_field=filter_field),
            item_stream):
        if keep:
            yield item
        # else:
        #     print(item[filter_field], )


def preprocess_files():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()
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
    # raw_dir = os.path.join(project_dir, 'data', 'raw')
    # # remove bots from labelers
    # bot_fpath = os.path.join(raw_dir, 'botnames.txt')
    # with open(bot_fpath, encoding='utf8') as f:
    #     botnames = set(i.strip() for i in f.read().split('\n'))
    # out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    # to_file(out_fpath, clean_items(item_stream=
    #                                filter(lambda item: re.findall(
    #                                    CONSPIRACY_THEORIST_RE,
    #                                    item['preprocessed_text'],
    #                                    flags=re.I | re.DOTALL | re.U | re.M),
    #                                       map(preprocess,
    #                                           filter(lambda x: x[
    #                                                                'author'] not in botnames,
    #                                                  stream_normalized_contribution(
    #                                                      fpath)))),
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

    # fpath = ct_sample_fpath
    # out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    # with Pool(40) as pool:
    #     to_file(out_fpath, clean_items(item_stream=
    #                                    pool.imap_unordered(preprocess,
    #                                                        stream_normalized_contribution(
    #                                                            fpath)),
    #                                    text_field='preprocessed_text',
    #                                    cleaned_text_field='processed_text',
    #                                    remove_punct=True, remove_digit=True,
    #                                    remove_stops=True,
    #                                    remove_pron=False,
    #                                    lemmatize=True, lowercase=True,
    #                                    n_process=-1
    #                                    ))
    #
    # fpath = default_sample_fpath
    # out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    # with Pool(40) as pool:
    #     to_file(out_fpath, clean_items(item_stream=
    #                                    pool.imap_unordered(preprocess,
    #                                                        stream_normalized_contribution(
    #                                                            fpath)),
    #                                    text_field='preprocessed_text',
    #                                    cleaned_text_field='processed_text',
    #                                    remove_punct=True, remove_digit=True,
    #                                    remove_stops=True,
    #                                    remove_pron=False,
    #                                    lemmatize=True, lowercase=True,
    #                                    n_process=-1
    #                                    ))

    # # read discussions filtered after preprocessing
    # discussion_id_fpath = os.path.join(interim_dir, 'discussion_ids.pkl')
    # if os.path.exists(discussion_id_fpath):
    #     logger.info(f'read existing discussions from {discussion_id_fpath}')
    #     with open(discussion_id_fpath, 'rb') as f:
    #         discussions = pickle.load(f)
    # else:
    #     fpath = labeling_fpath
    #     out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    #     logger.info(f'parse discussions from {out_fpath} into {discussion_id_fpath}')
    #     with open(out_fpath, encoding='utf8') as f:
    #         discussions = set(
    #             i['link_fullname'] for i in
    #             # i['name'] if ('selftext' in i) else i['link_id'] for i in
    #             map(json.loads, f))
    #     with open(discussion_id_fpath, 'wb+') as f:
    #         pickle.dump(discussions, f)
    # print(f'loaded {len(discussions)} discussion ids')
    # fpath = discussion_fpath
    # out_fpath = os.path.splitext(fpath)[0] + '_filtered.jsonl'
    # # preprocess only filtered discussions
    # to_file(out_fpath, filter_discussions(
    #     stream_normalized_contribution(
    #         fpath),
    #     discussions))

    # preprocess only filtered discussions
    logger.info('preprocess discussions')
    fpath = discussion_fpath
    out_fpath = os.path.splitext(fpath)[0] + '_filtered.jsonl'
    fpath = out_fpath
    out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'
    with Pool(40) as pool, open(fpath, encoding='utf8') as f:
        to_file(out_fpath, clean_items(item_stream=
                                       pool.imap_unordered(preprocess,
                                                           map(json.loads, f)),
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
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()
    # prepare input for the embeddings
    project_dir = Path(__file__).resolve().parents[2]

    interim_dir = os.path.join(project_dir, 'data', 'interim')

    # train embeddings
    os.makedirs(os.path.join(interim_dir, 'embeddings'), exist_ok=True)
    for dirname in os.listdir(os.path.join(interim_dir, 'text_years')):
        for fname in os.listdir(
                os.path.join(interim_dir, 'text_years', dirname)):

            fpath = os.path.join(interim_dir, 'text_years', dirname, fname)
            logger.info(f"training vectors for {fpath}")
            year = int(fpath[-len('.csv') - 4:-len('.csv')])
            try:
                Word2Vec.load(
                    os.path.join(interim_dir, 'embeddings', dirname,
                                 f"word2vec_{year}.model"))
                KeyedVectors.load(
                    os.path.join(interim_dir, 'embeddings', dirname,
                                 f"word2vec_{year}.wordvectors"))
                print(f'skipping: {fpath} already used for training')
            except Exception as e:
                print(f'cannot load vectors for {fpath}')
                print(e)
                corpus = MyCorpus(fpath)
                try:

                    model = Word2Vec(sentences=corpus, seed=42, epochs=10,
                                     workers=40)
                    os.makedirs(
                        os.path.join(interim_dir, 'embeddings', dirname),
                        exist_ok=True)

                    model.save(
                        os.path.join(interim_dir, 'embeddings', dirname,
                                     f"word2vec_{year}.model"))
                    word_vectors = model.wv
                    word_vectors.save(
                        os.path.join(interim_dir, 'embeddings', dirname,
                                     f"word2vec_{year}.wordvectors"))

                except RuntimeError as e:
                    print(e)


def separate_contributions_by_year():
    logger = logging.getLogger()
    # prepare input for the embeddings
    project_dir = Path(__file__).resolve().parents[2]

    interim_dir = os.path.join(project_dir, 'data', 'interim')

    labeling_fpath = os.path.join(interim_dir,
                                  'labeling_contributions_preprocessed_no_bot.jsonl')
    k = 100000
    sample_fpath = os.path.join(interim_dir,
                                f'sample_contributions_{k}_preprocessed.jsonl')
    ct_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                   f'sample_contributions_{k}_ct_preprocessed.jsonl')
    default_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                        f'sample_contributions_{k}_default_preprocessed.jsonl')
    discussion_fpath = os.path.join(interim_dir,
                                    'labeling_discussions_all_filtered_preprocessed_no_bot.jsonl')
    os.makedirs(os.path.join(interim_dir, 'text_years'), exist_ok=True)
    for folder_name, input_fpath in [
        ("labeling", labeling_fpath),
        #                              ("sample", sample_fpath),
        #                              ("ct_sample", ct_sample_fpath),
        #                              ("default_sample", default_sample_fpath),
        ("discussions", discussion_fpath)
    ]:
        out_fhandles = dict()
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

    for subsample, subreddits in [('ct', CONSPIRACY_SUBREDDITS),
                                  ('default', DEFAULT_SUBREDDITS)]:
        folder_name, input_fpath = (f"labeling_{subsample}", labeling_fpath)
        out_fhandles = dict()
        with open(input_fpath, encoding='utf8') as f:
            os.makedirs(os.path.join(interim_dir, 'text_years', folder_name),
                        exist_ok=True)
            logger.info(
                f"preparing {os.path.join(interim_dir, 'text_years', folder_name)}")
            for item in map(json.loads, f):
                if ('subreddit' in item) and (
                        item['subreddit'] not in subreddits): continue
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


def merge_samples_with_labeling_contributions():
    logger = logging.getLogger()
    # prepare input for the embeddings
    project_dir = Path(__file__).resolve().parents[2]

    interim_dir = os.path.join(project_dir, 'data', 'interim')
    for dirname in os.listdir(os.path.join(interim_dir, 'text_years')):
        if ('labeling' in dirname) or (
                'discussions' in dirname):  # don't re-inject labeling contributions in these cases (already present)
            continue
        os.makedirs(
            os.path.join(interim_dir, 'text_years', dirname + '_and_labeling'),
            exist_ok=True)
        for fname in os.listdir(
                os.path.join(interim_dir, 'text_years', dirname)):
            labeling_fpath = os.path.join(interim_dir, 'text_years',
                                          'labeling_ct' if dirname.startswith(
                                              'ct') else 'labeling_default',
                                          fname)
            regular_fpath = os.path.join(interim_dir, 'text_years', dirname,
                                         fname)
            out_fpath = os.path.join(interim_dir, 'text_years',
                                     dirname + '_and_labeling', fname)
            logger.info(f"merging labeling contribusions for {dirname}/{fname}")
            if not os.path.exists(labeling_fpath): continue
            with open(labeling_fpath, encoding='utf8') as f:
                labeling_contribs = list(f)
            with open(regular_fpath, encoding='utf8') as f:
                regular_contribs = list(f)
            contribs = regular_contribs + labeling_contribs
            random.shuffle(contribs)
            with open(out_fpath, "w+", encoding='utf8') as f:
                f.write('\n'.join(contribs))


def align_embeddings(max_year=2022, min_year=2012):
    logger = logging.getLogger()
    project_dir = Path(__file__).resolve().parents[2]

    interim_dir = os.path.join(project_dir, 'data', 'interim')
    embedding_dir = os.path.join(interim_dir, 'embeddings')
    aligned_embedding_dir = os.path.join(interim_dir, 'aligned_embeddings')
    for dirname in os.listdir(embedding_dir):
        align_years(in_dir=os.path.join(embedding_dir, dirname),
                    out_dir=os.path.join(aligned_embedding_dir, dirname),
                    max_year=max_year, min_year=min_year)


def enhance_with_perspective(max_retries=3,
                             requested_attributes=REQUESTED_ATTRIBUTES_ALL,
                             languages=['en']):
    logger = logging.getLogger()
    project_dir = Path(__file__).resolve().parents[2]

    interim_dir = os.path.join(project_dir, 'data', 'interim')

    labeling_fpath = os.path.join(interim_dir,
                                  'labeling_contributions_preprocessed_no_bot.jsonl')
    k = 100000
    sample_fpath = os.path.join(interim_dir,
                                f'sample_contributions_{k}_preprocessed.jsonl')
    ct_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                   f'sample_contributions_{k}_ct_preprocessed.jsonl')
    default_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                        f'sample_contributions_{k}_default_preprocessed.jsonl')
    discussion_fpath = os.path.join(interim_dir,
                                    'labeling_discussions_all_filtered_preprocessed_no_bot.jsonl')
    out_dir = os.path.join(interim_dir, 'perspective')
    os.makedirs(out_dir, exist_ok=True)

    # create the api connector and authenticate
    load_dotenv(find_dotenv())
    perspective_key = os.environ['PERSPECTIVE_KEY']
    service = discovery.build('commentanalyzer', 'v1alpha1',
                              developerKey=perspective_key,
                              discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                              static_discovery=False, )

    for input_fpath in [
        labeling_fpath,
        sample_fpath,
        ct_sample_fpath,
        default_sample_fpath,
        discussion_fpath,
    ]:
        output_fpath = os.path.join(out_dir,
                                    os.path.split(input_fpath)[:-1].replace(
                                        '.jsonl', '_perspective.jsonl'))
        with open(input_fpath, encoding='utf8') as f, open(output_fpath, 'w+',
                                                           encoding='utf8') as outf:
            perspectives = dict()
            for contribution in map(json.loads, f):
                fullname, text = contribution['fullname'], contribution['text']
                retries = 0
                done = False
                score = np.nan
                while (not done) and (retries < max_retries):
                    try:
                        score = get_toxicity_score(text, service,
                                                   requested_attributes,
                                                   languages)
                        done = True
                    except Exception as e:
                        if e.resp['status'] == '400':
                            score = -1
                            done = True
                            logger.info(e)
                        else:
                            logger.warning(e)
                            retries += 1
                            time.sleep(60)
                perspectives[fullname] = score
            for k, v in perspectives.items():
                outf.write(json.dumps({k: v}) + '\n')


def enhance_with_liwc(n_threads=4):
    logger = logging.getLogger()
    project_dir = Path(__file__).resolve().parents[2]

    interim_dir = os.path.join(project_dir, 'data', 'interim')

    labeling_fpath = os.path.join(interim_dir,
                                  'labeling_contributions_preprocessed_no_bot.jsonl')

    labeling_fpath = os.path.join(interim_dir,
                                  'labeling_contributions_preprocessed.jsonl')
    k = 100000
    sample_fpath = os.path.join(interim_dir,
                                f'sample_contributions_{k}_preprocessed.jsonl')
    ct_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                   f'sample_contributions_{k}_ct_preprocessed.jsonl')
    default_sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                        f'sample_contributions_{k}_default_preprocessed.jsonl')
    discussion_fpath = os.path.join(interim_dir,
                                    'labeling_discussions_all_filtered_preprocessed_no_bot.jsonl')
    out_dir = os.path.join(interim_dir, 'liwc')
    os.makedirs(out_dir, exist_ok=True)

    lexicon_path = os.path.join(project_dir, 'data', 'external',
                                'LIWC2015.jsonl')
    lexica = read_liwc(lexicon_path)
    matcher = get_matchers(lexica)

    for input_fpath in [
        labeling_fpath,
        # sample_fpath,
        # ct_sample_fpath,
        # default_sample_fpath,
        # discussion_fpath,
    ]:
        output_fpath = os.path.join(out_dir,
                                    os.path.split(input_fpath)[-1].replace(
                                        '.jsonl',
                                        '_liwc.jsonl'))
        with open(output_fpath, 'w+', encoding='utf8') as outf:
            pool = Pool(n_threads)
            for liwcs in tqdm(pool.imap_unordered(
                    partial(df_liwcifer, text_col='preprocessed_text',
                            matcher=matcher),
                    map(lambda chunk: chunk.set_index('fullname')[
                        ['preprocessed_text']],
                        pd.read_json(input_fpath, lines=True, chunksize=10000,
                                     encoding='utf8'))) ,
            desc=f'processing {output_fpath}'):
                for k, v in liwcs.to_dict(orient='index').items():
                    outf.write(json.dumps({k: v}) + '\n')

            # for chunk in pd.read_json(input_fpath, lines=True, chunksize=1000,
            #                           encoding='utf8'):
            #     liwcs = df_liwcifer(
            #         chunk.set_index('fullname')[['preprocessed_text']],
            #         'preprocessed_text', matcher)
            #
            #     for k, v in liwcs.to_dict(orient='index').items():
            #         outf.write(json.dumps({k: v}) + '\n')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # preprocess_files()
    # # should run the notebook to find bots
    # # then, should run the filter_bots function in make_dataset
    # separate_contributions_by_year()
    # merge_samples_with_labeling_contributions()
    # build_embeddings()
    # align_embeddings()
    # enhance_with_perspective()
    enhance_with_liwc()
