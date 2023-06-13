# -*- coding: utf-8 -*-
import datetime
import json
import os.path
import re
from collections import defaultdict
from json import JSONDecodeError

import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import zstandard as zstd
import io

from zstandard import ZstdError

from src.data.collect_reddit import search_pushshift

CONSPIRACY_THEORIST_RE = '(conspiracist)|(conspiracy theorist)'
CONSPIRACY_SUBREDDITS = ["984isreality", "911truth", "actualconspiracies", "Bilderberg", "C_S_T", "CHEMPRINTS",
                         "Chemtrail", "chemtrails", "chrisolivertimes", "climateskeptics", "ClintonInvestigation",
                         "conspiracies", "conspiracy", "conspiracy_commons", "conspiracydocumentary", "conspiracyfact",
                         "conspiracyhub", "ConspiracyII", "ConspiracyPublic", "conspiracytheories", "ConspiracyTheory",
                         "conspiracyundone", "ConspiracyX", "ConspiracyZone", "conspiro", "CorporateMalfeasance",
                         "CosmicDisclosure", "DescentIntoTyranny", "DigitalCartel", "FalseFlagWatch",
                         "finlandConspiracy", "FringeTheory", "HealthConspiracy", "highersidechats", "HOLLOWEARTH",
                         "LimitedHangouts", "moonhoax", "notaglobe", "OccultConspiracy", "OccupyLangley", "PedoGate",
                         "PoliticalConspiracy", "ProConspiracy", "ProjectSTARGATE", "reptiliandude", "reptilians",
                         "RomeRules", "TargetedEnergyWeapons", "TargetedIndividuals", "theworldisflat", "TopConspiracy",
                         "TruthLeaks", "UNAgenda21", "VaccinesCause", "WhereIsAssange"]
DEFAULT_SUBREDDITS = ["AskReddit", "announcements", "funny", "pics", "todayilearned", "science", "IAmA", "blog",
                      "videos", "worldnews", "gaming", "movies", "Music", "aww", "news", "gifs", "askscience",
                      "explainlikeimfive", "EarthPorn", "books", "television", "LifeProTips", "sports", "DIY",
                      "Showerthoughts", "space", "Jokes", "tifu", "food", "photoshopbattles", "Art",
                      "InternetIsBeautiful", "mildlyinteresting", "GetMotivated", "history", "nottheonion", "gadgets",
                      "dataisbeautiful", "Futurology", "Documentaries", "listentothis", "personalfinance", "philosophy",
                      "nosleep", "creepy", "OldSchoolCool", "UpliftingNews", "WritingPrompts", "TwoXChromosomes"]


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath, text_field='body'):
    """ finds contribusions that mention the term "conspiracy theorist(s)/conspiracist(s)"
    """
    logger = logging.getLogger(__name__)
    logger.info('search labeling instances')
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
    try:
        yield from io.TextIOWrapper(reader, encoding='utf-8')
    except ZstdError as e:
        print('error reading')
        print(e)


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
        if not os.path.exists(outfile):
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


def consolidate_files(input_dir, output_fpath, file_suffix):
    with open(output_fpath, 'w+', encoding='utf8') as f:
        for infpath in os.listdir(input_dir):
            if infpath.endswith(file_suffix):
                contribution_prefix = "t3_" if 'RS' in infpath else 't1_'
                with open(os.path.join(input_dir, infpath), encoding='utf8') as inf:
                    for l in map(json.loads, inf):
                        if 'name' not in l:
                            l['name'] = contribution_prefix + l['id']
                        f.write(json.dumps(l) + '\n')


def filter_discussions_(args):
    return filter_discussions(*args)


def filter_discussions(input_filepath, output_filepath, filter_field,
                       filter_values):
    logger = logging.getLogger(__name__)
    logger.info('filtering discussions')
    logger.info(f'{input_filepath} to {output_filepath}')
    with open(output_filepath, 'a+', encoding='utf8') as f:
        for contribution in read_zst(input_filepath):
            if ('selftext' in contribution) and ('name' not in contribution):
                contribution['name'] = 't3_' + contribution['id']

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
            i['name'] if ('selftext' in i) else i['link_id'] for i in
            map(json.loads, f))
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    contribution_fpaths = get_contribution_fpaths()

    with Pool(40) as pool:
        args = args_builder_discussions(contribution_fpaths, output_dir,
                                        output_suffix, discussions)
        pool.map(filter_discussions_, args)


def filter_authors_(args):
    return filter_authors(*args)


def filter_authors(input_filepath, output_filepath,
                   filter_values):
    logger = logging.getLogger(__name__)
    logger.info('filtering authors')
    logger.info(f'{input_filepath} to {output_filepath}')
    with open(output_filepath, 'a+', encoding='utf8') as f:
        for contribution in read_zst(input_filepath):
            if (not contribution) or ('author' not in contribution): continue
            if ('selftext' in contribution) and ('name' not in contribution):
                contribution['name'] = 't3_' + contribution['id']

            if contribution['author'] in filter_values:
                f.write(json.dumps(contribution) + '\n')


def args_builder_authors(contribution_fpaths, output_dir, output_suffix,
                         filter_values):
    args = list()
    for infile in contribution_fpaths:
        outfile = os.path.split(infile)[-1][:-len('.zst')] + output_suffix
        outfile = os.path.join(output_dir, outfile)
        args.append((infile, outfile, filter_values))
    return args


def collect_authors(input_fpath, bot_fpath, output_dir,  # author_fpath,
                    output_suffix='_labelers.jsonl'):
    with open(input_fpath, encoding='utf8') as f:
        authors = set(
            i['author'] for i in
            map(json.loads, f))

    with open(bot_fpath, encoding='utf8') as f:
        botnames = set(i.strip() for i in f.read().split('\n'))
    authors = {author for author in authors if author not in botnames}
    authors = authors.difference({'[deleted]', '[removed]'})
    # with open(author_fpath, 'w+', encoding='utf8') as f:
    #     f.write('\n'.join(authors))
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    contribution_fpaths = get_contribution_fpaths()

    with Pool(40) as pool:
        args = args_builder_authors(contribution_fpaths, output_dir,
                                    output_suffix, authors)
        pool.map(filter_authors_, args)


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


def sample_instances(input_filepath, output_filepath, k, subreddits=None):
    logger = logging.getLogger(__name__)
    logger.info('sampling random contributions')
    logger.info(f'{input_filepath} to {output_filepath}')
    algo = AlgorithmL(k)
    for contribution in read_zst(input_filepath):
        try:
            if subreddits and (contribution['subreddit'] not in subreddits):
                continue
            else:
                algo.add(contribution)
        except KeyError:
            # logger.error('no subreddit specified')
            # logger.error(str(contribution))
            pass
    with open(output_filepath, 'a+', encoding='utf8') as f:
        for contribution in algo.reservoir:
            f.write(json.dumps(contribution) + '\n')


def sample_instances_(args):
    return sample_instances(*args)


def sample_contributions(k, output_dir,
                         output_suffix='_sample.jsonl', subreddits=None):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    contribution_fpaths = get_contribution_fpaths()

    args = list()
    for infile in contribution_fpaths:
        outfile = os.path.split(infile)[-1][:-len('.zst')] + output_suffix
        outfile = os.path.join(output_dir, outfile)
        args.append((infile, outfile, k, subreddits))

    with Pool(40) as pool:
        pool.map(sample_instances_, args)


def filter_bots():
    logger = logging.getLogger(__name__)
    logger.info('filtering bots')

    project_dir = Path(__file__).resolve().parents[2]
    interim_dir = os.path.join(project_dir, 'data', 'interim')
    raw_dir = os.path.join(project_dir, 'data', 'raw')

    # remove remaining bot authors
    logger.info('reading bots')
    with open(os.path.join(raw_dir, 'botnames_expanded.txt'), encoding='utf8') as f:
        botnames = set(i.strip() for i in f)

    # filter labeling contributions
    no_bot_suffix = '_no_bot.'
    # find discussions from labeling contributions while at it
    labeling_discussions = set()
    labeling_fpath = os.path.join(interim_dir,
                                  'labeling_contributions_preprocessed.jsonl')
    logger.info('filtering labeling contributions')
    with open(labeling_fpath, encoding='utf8') as f, \
            open(labeling_fpath.replace('.', no_bot_suffix), 'w+', encoding='utf8') as fout:
        for line in f:
            contribution = json.loads(line)
            if contribution.get('author', '') not in botnames:
                fout.write(line)
                labeling_discussions.add(contribution['link_fullname'])

    # filter labeling discussions
    logger.info('filtering discussions')
    discussion_fpath = os.path.join(interim_dir,
                                    'labeling_discussions_all_filtered_preprocessed.jsonl')
    with open(discussion_fpath, encoding='utf8') as f, \
            open(discussion_fpath.replace('.', no_bot_suffix), 'w+', encoding='utf8') as fout:
        for line in f:
            contribution = json.loads(line)
            if contribution['link_fullname'] in labeling_discussions:
                fout.write(line)


def divide_discussions(in_fpath, subreddit_subsets={'ct': CONSPIRACY_SUBREDDITS,
                                                    'default': DEFAULT_SUBREDDITS}):
    destinations = [(infix, subreddits, open(in_fpath.replace('preprocessed', f'preprocessed_{infix}'),
                                             'w+', encoding='utf8'))
                    for infix, subreddits in subreddit_subsets.items()]
    with open(in_fpath, encoding='utf8') as f:
        for contribution in map(json.loads, f):
            for infix, subreddits, outf in destinations:
                if contribution.get('subreddit', None) in subreddits:
                    outf.write(json.dumps(contribution) + '\n')
    for _, _, outf in destinations:
        outf.close()


def subsample_further(in_dir, n_per_mo=10000):
    logger = logging.getLogger()

    def dt_to_months_since(created_utc):
        d = datetime.datetime.fromtimestamp(float(created_utc))
        return d.year, d.month

    for fname in os.listdir(in_dir):
        if not (fname.startswith('sample_contribution') and fname.endswith('preprocessed.jsonl')):
            continue
        logger.info(f'processing {fname}')
        algos = defaultdict(lambda: AlgorithmL(n_per_mo))
        with open(os.path.join(in_dir, fname), encoding='utf8') as f:
            for contribution in map(json.loads, f):
                algos[dt_to_months_since(contribution['created_utc'])].add(contribution)
        logger.info(f'read {fname}')
        out_fname = os.path.join(in_dir, fname.replace('100000', f'{n_per_mo}'))
        with open(out_fname, 'w+', encoding='utf8') as outf:
            for _, algo in sorted(algos.items(), key=lambda x: x[0]):
                for contribution in algo.reservoir:
                    outf.write(json.dumps(contribution, sort_keys=True) + '\n')


def fullname2int(fullname):
    if fullname is None:
        return fullname
    return int(fullname.split('_')[-1], 36)


def extract_thread_structure(labeling_fpath, discussions_fpath, out_fpath):
    # read all labeling instances
    # keep just fullnames
    with open(labeling_fpath, encoding='utf8') as f:
        labeling_fullnames = set(map(fullname2int, map(lambda x: json.loads(x)['fullname'], f)))

    # first pass
    keys_of_interest = {"fullname", "link_fullname", "parent_fullname", "created_utc", "subreddit"}
    # stream discussions
    discussions = defaultdict(list)
    with open(discussions_fpath, encoding='utf8') as f:
        for contribution in map(lambda x: json.loads(x), f):
            # keep metadata: fullname, link_fullname, parent_fullname, created_utc, subreddit
            contribution = {k: v for k, v in contribution.items() if k in keys_of_interest}
            # string to int for fullnames
            for k in contribution.keys():
                if 'fullname' in k:
                    contribution[k] = fullname2int(contribution[k])
            # create metadata: is_labeling
            contribution['is_labeling'] = contribution['fullname'] in labeling_fullnames
            discussion = contribution.pop('link_fullname')
            discussions[discussion].append(contribution)
    # all in memory and then dump to file? separate discussion?
    with open(out_fpath, 'w+') as f:
        for item in discussions.items():
            f.write(json.dumps(item, sort_keys=True)+'\n')

    # second pass
    # create metadata:
    # thread_size (short threads wouldn't help much),
    # is_first_labeling (if multiple labeling instances, may want to only keep the first),
    # labeling_size,
    # index_from_first_labeling
    # timedelta_from_first_labeling
    # filter: on size, on +-index_from_first_labeling, +-timedelta_from_first_labeling, same but only in induced subgraph
    pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    interim_dir = os.path.join(project_dir, 'data', 'interim')
    raw_dir = os.path.join(project_dir, 'data', 'raw')

    # parse_files(os.path.join(project_dir, 'data', 'interim'))
    # labeling_fpath = os.path.join(project_dir, 'data', 'interim',
    #                               'labeling_contributions.jsonl')
    # consolidate_files(interim_dir,
    #                   labeling_fpath,
    #                   file_suffix='_labeling.jsonl')
    #
    # discussion_suffix = '_discussions.jsonl'
    # collect_discussions(labeling_fpath, interim_dir,
    #                     output_suffix=discussion_suffix)
    #
    # discussion_fpath = os.path.join(project_dir, 'data', 'interim',
    #                                 'labeling_discussions_all.jsonl')
    # consolidate_files(interim_dir,
    #                   discussion_fpath,
    #                   file_suffix=discussion_suffix)
    #
    # sample_suffix = '_sample.jsonl'
    # k = 100000
    # sample_contributions(k=k, output_dir=interim_dir, output_suffix=sample_suffix)
    # sample_fpath = os.path.join(project_dir, 'data', 'interim',
    #                             f'sample_contributions_{k}.jsonl')
    # consolidate_files(interim_dir,
    #                   sample_fpath,
    #                   file_suffix=sample_suffix)

    # sample_suffix = '_sample_ct.jsonl'
    # k = 100000
    # sample_contributions(k=k, output_dir=interim_dir, output_suffix=sample_suffix, subreddits=CONSPIRACY_SUBREDDITS)
    # sample_fpath = os.path.join(project_dir, 'data', 'interim',
    #                             f'sample_contributions_{k}_ct.jsonl')
    # consolidate_files(interim_dir,
    #                   sample_fpath,
    #                   file_suffix=sample_suffix)

    #
    # sample_suffix = '_sample_default.jsonl'
    # k = 100000
    # sample_contributions(k=k, output_dir=interim_dir, output_suffix=sample_suffix, subreddits=DEFAULT_SUBREDDITS)
    # sample_fpath = os.path.join(project_dir, 'data', 'interim',
    #                             f'sample_contributions_{k}_default.jsonl')
    # consolidate_files(interim_dir,
    #                   sample_fpath,
    #                   file_suffix=sample_suffix)
    #
    # filter_bots()

    # labeler_suffix = '_labelers.jsonl'
    #
    # filtered_authors_fpath = os.path.join(interim_dir, 'labeling_contributions_preprocessed_no_bot.jsonl')
    # collect_authors(input_fpath=filtered_authors_fpath,
    #                 bot_fpath=os.path.join(raw_dir, 'botnames_expanded.txt'),
    #                 output_dir=interim_dir,
    #                 output_suffix=labeler_suffix)
    # labeler_fpath = os.path.join(project_dir, 'data', 'interim',
    #                             f'labelers_all.jsonl')
    # consolidate_files(interim_dir,
    #                   labeler_fpath,
    #                   file_suffix=labeler_suffix)

    # divide_discussions(os.path.join(interim_dir, "labeling_discussions_all_filtered_preprocessed_no_bot.jsonl"))

    # divide_discussions(os.path.join(interim_dir, "labeling_discussions_all_filtered_preprocessed_no_bot.jsonl"),
    #                    subreddit_subsets={'default': DEFAULT_SUBREDDITS})
    # subsample_further(interim_dir)

    extract_thread_structure(labeling_fpath=os.path.join(interim_dir, 'labeling_contributions_preprocessed_no_bot.jsonl'),
                             discussions_fpath=os.path.join(interim_dir, "labeling_discussions_all_filtered_preprocessed_no_bot.jsonl"),
                             out_fpath=os.path.join(interim_dir, 'thread_structres.jsonl'))