# -*- coding: utf-8 -*-
import json
import os.path
import re
from json import JSONDecodeError

import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
import zstandard as zstd
import io

from src.data.collect_reddit import search_pushshift

CONSPIRACY_THEORIST_RE = '(conspiracist)|(conspiracy theorist)'


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath, text_field='body'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    logger.info(f'{input_filepath} to {output_filepath}')
    with open(output_filepath, 'a+', encoding='utf8') as f:
        for contribution in read_zst(input_filepath):
            text = contribution[text_field]
            if not text: continue
            if re.findall(CONSPIRACY_THEORIST_RE, text,
                          flags=re.I | re.DOTALL | re.U | re.M):
                f.write(json.dumps(contribution) + '\n')


def decompress(fh):
    reader = zstd.ZstdDecompressor(max_window_size=2147483648).stream_reader(fh)
    yield from io.TextIOWrapper(reader, encoding='utf-8')


def read_zst(fpath):
    with open(fpath, 'rb') as fh:
        for line in decompress(fh):
            try:
                yield json.loads(line)
            except JSONDecodeError as e:
                print(f"error in {fpath}")
                print(e)


def main_(args):
    return main(*args)


def args_builder_textfield(contribution_fpaths, output_dir, output_suffix):
    args = list()
    for infile in contribution_fpaths:
        outfile = os.path.split(infile)[-1][:-len('.zst')] + output_suffix
        outfile = os.path.join(output_dir, outfile)
        text_field = "selftext" if "RS" in infile else "body"
        args.append((infile, outfile, text_field))
    return args


def parse_files(output_dir, process_func=main_, output_suffix='_labeling.jsonl',
                args_builder=args_builder_textfield):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    contribution_fpaths = get_contribution_fpaths()

    with Pool(40) as pool:
        args = args_builder(contribution_fpaths, output_dir, output_suffix)
        pool.map(process_func, args)
        pool.join()


def consolidate_files(input_dir, output_fpath, file_suffix):
    with open(output_fpath, 'w+', encoding='utf8') as f:
        for infpath in os.listdir(input_dir):
            if infpath.endswith(file_suffix):
                with open(os.path.join(input_dir, infpath), encoding='utf8') as inf:
                    for l in inf:
                        f.write(l)


def filter_instances_(args):
    return filter_instances(*args)


def filter_instances(input_filepath, output_filepath, filter_field,
                     filter_values):

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    logger.info(f'{input_filepath} to {output_filepath}')
    with open(output_filepath, 'a+', encoding='utf8') as f:
        for contribution in read_zst(input_filepath):

            if contribution[filter_field] in filter_values:
                f.write(json.dumps(contribution) + '\n')


def args_builder_discussions(contribution_fpaths, output_dir, output_suffix,
                             filter_values):
    args = list()
    for infile in contribution_fpaths:
        outfile = os.path.split(infile)[-1][:-len('.zst')] + output_suffix
        outfile = os.path.join(output_dir, outfile)
        filter_field = "name" if "RS" in infile else "link_id"
        args.append((infile, outfile, filter_field, filter_values))
    return args


def collect_discussions(input_fpath, output_dir,
                        output_suffix='_discussions.jsonl'):
    with open(input_fpath, encoding='utf8') as f:
        discussions = set(
            i['name'] if i['name'].startswith('t3_') else i['link_id'] for i in
            map(json.loads, f))

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    contribution_fpaths = get_contribution_fpaths()

    with Pool(40) as pool:
        args = args_builder_discussions(contribution_fpaths, output_dir,
                                        output_suffix, discussions)
        pool.map(filter_instances_, args)
        pool.join()


def get_contribution_fpaths():
    load_dotenv(find_dotenv())
    pushshift_dir = os.environ['PUSHSHIFT_DIR']
    contribution_fpaths = [os.path.join(pushshift_dir, fname)
                           for fname in [os.path.join(f'{folder}',
                                                      f'{contribution}_{year}-{month:02}.zst')
                                         for (contribution, folder) in
                                         (('RS', 'submissions'),
                                          ('RC', 'comments')
                                          )
                                         for year in range(2005, 2024)
                                         for month in range(1, 13)]]
    contribution_fpaths = [i for i in contribution_fpaths if os.path.exists(i)]
    return contribution_fpaths


def reservoir_sample(instream, k):
    # reservoir sample algorithm l
    reservoir = list()
    j = 0
    try:
        while j < k:
            reservoir.append(next(instream))
            j += 1
    except StopIteration:
        return reservoir

    w = np.exp(np.log(np.random.random()) / k)
    i = j
    for j, element in enumerate(instream, start=j):
        if j == i:
            reservoir[np.random.randint(0, k)] = element
            i += (np.floor(np.log(np.random.random()) / np.log(1 - w))) + 1
            w *= np.exp(np.log(np.random.random()) / k)
        else:
            pass
    return reservoir


class AlgorithmL:
    # ported from https://richardstartin.github.io/posts/reservoir-sampling#algorithm-l
    def __init__(self, k):
        self.reservoir = list()
        self.k = k
        self.next = k
        self.counter = 0
        self.w = np.exp(np.log(np.random.random()) / k)
        self.skip()

    def add(self, item):
        if self.counter < self.k:
            self.reservoir.append(item)
        else:
            if self.counter == self.next:
                self.reservoir[np.random.randint(0, self.k)] = item
                self.skip()
        self.counter += 1

    def skip(self):
        self.next += (np.floor(
            np.log(np.random.random()) / np.log(1 - self.w))) + 1
        self.w *= np.exp(np.log(np.random.random()) / self.k)

def sample_instances(input_filepath, output_filepath, k):

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    logger.info(f'{input_filepath} to {output_filepath}')
    algo=AlgorithmL(k)
    for contribution in read_zst(input_filepath):
        algo.add(contribution)
    with open(output_filepath, 'a+', encoding='utf8') as f:
        for contribution in algo.reservoir:
            f.write(json.dumps(contribution) + '\n')

def sample_instances_(args):
    return sample_instances(*args)
def sample_contributions(k, output_dir,
                        output_suffix='_sample.jsonl'):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    contribution_fpaths = get_contribution_fpaths()

    args = list()
    for infile in contribution_fpaths:
        outfile = os.path.split(infile)[-1][:-len('.zst')] + output_suffix
        outfile = os.path.join(output_dir, outfile)
        args.append((infile, outfile, k))

    with Pool(40) as pool:
        pool.map(sample_instances_, args)
        pool.join()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    # infile = os.path.join(project_dir, 'data', 'external', 'RC_2009-09.zst')
    # outfile = os.path.join(project_dir, 'data', 'interim', 'RC_2009-09.jsonl')
    # main(input_filepath=infile, output_filepath=outfile)

    # parse_files(os.path.join(project_dir, 'data', 'interim'))
    interim_dir = os.path.join(project_dir, 'data', 'interim')
    labeling_fpath = os.path.join(project_dir, 'data', 'interim',
                                  'labeling_contributions.jsonl')
    consolidate_files(interim_dir,
                      labeling_fpath,
                      file_suffix='_labeling.jsonl')

    discussion_suffix = '_discussions.jsonl'
    collect_discussions(labeling_fpath, interim_dir,
                        output_suffix=discussion_suffix)

    discussion_fpath = os.path.join(project_dir, 'data', 'interim',
                                    'labeling_discussions.jsonl')
    consolidate_files(interim_dir,
                      discussion_fpath,
                      file_suffix=discussion_suffix)

    sample_suffix = '_sample.jsonl'
    k = 100000
    sample_contributions(k=k, output_dir=interim_dir, output_suffix=sample_suffix)
    sample_fpath = os.path.join(project_dir, 'data', 'interim',
                                    f'sample_contributions_{k}.jsonl')
    consolidate_files(interim_dir,
                      sample_fpath,
                      file_suffix=sample_suffix)

    # outfile = os.path.join(project_dir, 'data', 'interim',
    #                        'labeling_contributions.jsonl')
    # search_pushshift(store_path=outfile,
    #                  q='''conspiracist|conspiracists|"conspiracy theorist"|"conspiracy theorists"''')
