# -*- coding: utf-8 -*-
import datetime
import json
import os.path
import re
import time
from collections import defaultdict, Counter
from functools import partial
from json import JSONDecodeError

import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import zstandard as zstd
import io
import networkx as nx
from scipy.stats import zscore, gzscore

from zstandard import ZstdError

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from src.data.collect_reddit import search_pushshift
from src.utils import chunkize_iter

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


def int2link_fullname(fullname_int):
    return f"t3_{np.base_repr(fullname_int)}"


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
            f.write(json.dumps(item, sort_keys=True) + '\n')


def filter_threads(in_fpath, seconds_delta, index_delta, min_thread_size, out_folder):
    logger = logging.getLogger()
    # second pass
    os.makedirs(out_folder, exist_ok=True)
    with open(in_fpath) as f, open(os.path.join(out_folder, 'discussions_by_size.jsonl'), 'w+') as outf_size, \
            open(os.path.join(out_folder, 'discussions_by_index_delta.jsonl'), 'w+') as outf_index_delta, \
            open(os.path.join(out_folder, 'discussions_by_time_delta.jsonl'), 'w+') as outf_time_delta, \
            open(os.path.join(out_folder, 'discussions_by_index_delta_subthread.jsonl'),
                 'w+') as outf_index_delta_subthread, \
            open(os.path.join(out_folder, 'discussions_by_time_delta_subthread.jsonl'),
                 'w+') as outf_time_delta_subthread, \
            open(os.path.join(out_folder, 'discussions_by_subthread.jsonl'),
                 'w+') as outf_subthread:

        for link_fullname, thread in map(json.loads, f):
            for contribution in thread:
                contribution['created_utc'] = float(contribution['created_utc'])
            thread = sorted(thread, key=lambda x: x['created_utc'], )
            # create metadata:
            # - thread_size (short threads wouldn't help much),
            thread_size = len(thread)

            labeling_contribution_indices, labeling_contributions = zip(*filter(lambda x: x[1]['is_labeling'],
                                                                                enumerate(thread)))
            # # - labeling_size,
            # labeling_size = len(labeling_contribution_indices)
            # - index
            # - is_first_labeling (if multiple labeling instances, may want to only keep the first),
            for i in range(thread_size):
                thread[i]['index'] = i
                thread[i]['is_first_labeling'] = i == labeling_contribution_indices[0]

            for labeling_index in labeling_contribution_indices:
                # get sub-thread by combining ancestors and descendants from labeling instance
                G = nx.DiGraph()
                G.add_nodes_from(contribution['fullname'] for contribution in thread)
                G.add_edges_from(
                    (contribution['fullname'], contribution['parent_fullname']) for contribution in thread if
                    contribution['parent_fullname'])
                labeling_fullname = thread[labeling_index]['fullname']
                if labeling_fullname is None:
                    logger.error('no fullname for ' + json.dumps(thread[labeling_index]))
                    continue
                if not G.has_node(labeling_fullname):
                    logger.error(f'G {G.number_of_nodes()} {G.number_of_edges()} disconnected: ' + json.dumps(
                        thread[labeling_index]))
                    logger.error(G.nodes)
                    raise ValueError
                ancestors = list(nx.ancestors(G, labeling_fullname))
                descendants = list(nx.descendants(G, labeling_fullname))
                connected_contribution_fullnames = set(ancestors)
                connected_contribution_fullnames.update(descendants)
                connected_contributions = [i for i in thread if i['fullname'] in connected_contribution_fullnames]
                subthread = sorted([thread[labeling_index]] + connected_contributions,
                                   key=lambda x: x['created_utc'])
                labeling_index_subthread = None
                for i, contribution in enumerate(subthread):
                    if contribution['fullname'] == labeling_fullname:
                        labeling_index_subthread = i
                        break

                # filter: on size, on +-index_from_labeling, +-timedelta_from_labeling
                by_index_slice = [contribution for contribution in thread if abs(
                    contribution['index'] - thread[labeling_index]['index']) < index_delta]
                by_time_slice = [contribution for contribution in thread if abs(
                    contribution['created_utc'] - thread[labeling_index][
                        'created_utc']) < seconds_delta]
                # filter: same but only in induced subgraph
                by_index_subthread_slice = [contribution for contribution_index_subthread, contribution in
                                            enumerate(subthread) if abs(
                        contribution_index_subthread - labeling_index_subthread) < index_delta]
                by_time_subthread_slice = [contribution for contribution in subthread if abs(
                    contribution['created_utc'] - thread[labeling_index][
                        'created_utc']) < seconds_delta]

                # persist
                outf_index_delta.write(json.dumps({link_fullname: by_index_slice}, sort_keys=True) + '\n')
                outf_time_delta.write(json.dumps({link_fullname: by_time_slice}, sort_keys=True) + '\n')
                outf_index_delta_subthread.write(
                    json.dumps({link_fullname: by_index_subthread_slice}, sort_keys=True) + '\n')
                outf_time_delta_subthread.write(
                    json.dumps({link_fullname: by_time_subthread_slice}, sort_keys=True) + '\n')
                outf_subthread.write(json.dumps({link_fullname: subthread}, sort_keys=True) + '\n')
                if thread_size >= min_thread_size:
                    outf_size.write(json.dumps({link_fullname: thread}, sort_keys=True) + '\n')


def consolidate_filtered_threads(discussions_fpath, filtered_ids_fpath, out_fpath_all, out_fpath_default, out_fpath_ct):
    logger = logging.getLogger()
    with open(discussions_fpath, encoding='utf8') as inf, \
            open(filtered_ids_fpath) as filterf, \
            open(out_fpath_all, 'w+', encoding='utf8') as outf_all, \
            open(out_fpath_ct, 'w+', encoding='utf8') as outf_ct, \
            open(out_fpath_default, 'w+', encoding='utf8') as outf_default:
        filter_discussions = defaultdict(set)
        for thread_item in map(json.loads, filterf):
            for k, vv in thread_item.items():
                # necessary as there may be multiple lines corrensponding to the same discussion, related to different labeling instances
                filter_discussions[int(k)].update({v['fullname'] for v in vv})
        for contribution in map(json.loads, inf):
            filter_contribution_fullnames = filter_discussions.get(fullname2int(contribution['link_fullname']), [])
            if fullname2int(contribution['fullname']) in filter_contribution_fullnames:
                out_line = json.dumps(contribution, sort_keys=True) + '\n'
                outf_all.write(out_line)
                contribution_subreddit = contribution.get('subreddit', None)
                if contribution_subreddit is None:
                    logger.warning(f'no subreddit for {json.dumps(contribution)}')
                if contribution_subreddit in CONSPIRACY_SUBREDDITS:
                    outf_ct.write(out_line)
                elif contribution_subreddit in DEFAULT_SUBREDDITS:
                    outf_default.write(out_line)


def compute_baseline_volume_(in_fpath, out_folder='counts'):
    logger = logging.getLogger()

    cntr = Counter()
    cntr_ct = Counter()
    cntr_default = Counter()

    in_fname = os.path.basename(in_fpath)
    out_fpath = os.path.join(out_folder, in_fname.replace('.zst', '_all_counts.json'))
    out_ct_fpath = os.path.join(out_folder, in_fname.replace('.zst', '_ct_counts.json'))
    out_default_fpath = os.path.join(out_folder, in_fname.replace('.zst', '_default_counts.json'))
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)

    logger.info(f'processing {in_fname}')
    for contribution in read_zst(in_fpath):
        timestamp = contribution.get('created_utc', None)
        if timestamp:
            timestamp = datetime.datetime.fromtimestamp(float(timestamp))
            truncated_timestamp = time.mktime(timestamp.date().timetuple())
            cntr[truncated_timestamp] += 1
            if contribution.get('subreddit', None) in CONSPIRACY_SUBREDDITS:
                cntr_ct[truncated_timestamp] += 1
            elif contribution.get('subreddit', None) in DEFAULT_SUBREDDITS:
                cntr_default[truncated_timestamp] += 1
    logger.info(f'writing to {out_fpath}')
    with open(out_fpath, 'w+') as f:
        f.write(json.dumps(cntr, sort_keys=True))
    with open(out_ct_fpath, 'w+') as f:
        f.write(json.dumps(cntr_ct, sort_keys=True))
    with open(out_default_fpath, 'w+') as f:
        f.write(json.dumps(cntr_default, sort_keys=True))


def compute_baseline_volume(out_folder):
    fpaths = get_contribution_fpaths()
    with Pool(40) as pool:
        pool.map(partial(compute_baseline_volume_, out_folder=out_folder), fpaths)


def consolidate_baseline_volume(in_folder):
    all_counts = Counter()
    ct_counts = Counter()
    default_counts = Counter()
    for fname in filter(lambda x: x.startswith('RC') or x.startswith('RS'),
                        os.listdir(in_folder)):
        fpath = os.path.join(in_folder, fname)
        with open(fpath) as f:
            cnts = json.load(f)
            if 'ct' in fname:
                ct_counts += cnts
            elif 'default' in fname:
                default_counts += cnts
            elif 'all' in fname:
                all_counts += cnts
    with open(os.path.join(in_folder, 'all_counts.json'), 'w+') as f:
        json.dump(all_counts, f)
    with open(os.path.join(in_folder, 'ct_counts.json'), 'w+') as f:
        json.dump(ct_counts, f)
    with open(os.path.join(in_folder, 'default_counts.json'), 'w+') as f:
        json.dump(default_counts, f)


def labeler_subreddit_distribution(labeling_fpath, labeler_contributions_fpath, fpath_histogram_before,
                                   fpath_histogram_after):
    with open(labeling_fpath, encoding='utf8') as labeling_f, open(labeler_contributions_fpath,
                                                                   encoding='utf8') as labeler_f:
        # compute the first time a labeler labels
        labeler_tholds = defaultdict(set)
        for labeling_instance in map(json.loads, labeling_f):
            labeler_tholds[labeling_instance['author']].add(float(labeling_instance['created_utc']))
        labeler_tholds = {k: min(v) for k, v in labeler_tholds.items()}

        # compute how many times did each labeler contribute to a subreddit, before and after their first labeling instance
        labeler_histograms_before = defaultdict(Counter)
        labeler_histograms_after = defaultdict(Counter)
        for contribution in map(json.loads, labeler_f):
            created_utc = float(contribution['created_utc'])
            labeler = contribution['author']
            subreddit = contribution.get('subreddit', None)
            if (labeler not in labeler_tholds) or (subreddit is None):
                continue
            if created_utc < labeler_tholds[labeler]:
                labeler_histograms_before[labeler][subreddit] += 1
            else:
                labeler_histograms_after[labeler][subreddit] += 1
    with open(fpath_histogram_before, 'w+', encoding='utf8') as f:
        for k, v in labeler_histograms_before.items():
            f.write(json.dumps({k: v}, sort_keys=True) + '\n')
    with open(fpath_histogram_after, 'w+', encoding='utf8') as f:
        for k, v in labeler_histograms_after.items():
            f.write(json.dumps({k: v}, sort_keys=True) + '\n')


def _process_chunk(chunk, ct, dims, min_subreddits_per_user):
    most_frequent_subs = dict()
    subreddit_sums = dict()
    outf_dims = list()
    outf_ct = list()
    for line in chunk:
        user, data = tuple(line.items())[0]
        max_count = 0
        max_subreddit = None
        total_count = 0
        for subreddit, count in data.items():
            if count > max_count:
                max_count = count
                max_subreddit = subreddit
            total_count += count
            if len(data) > min_subreddits_per_user:
                subreddit_sums[subreddit] = subreddit_sums.get(subreddit, 0) + 1
        most_frequent_subs[user] = max_subreddit
        normed = {subreddit: count / total_count for subreddit, count in data.items()}
        outf_dims.append(json.dumps({user: {dim: sum(frac * dims.loc[subreddit, dim]
                                                     for subreddit, frac in normed.items()
                                                     if subreddit in dims.index)
                                            for dim in dims.columns
                                            }},
                                    sort_keys=True) + '\n')
        outf_ct.append(json.dumps({user: sum(frac * ct.loc[subreddit, 'conspiracy']
                                             for subreddit, frac in normed.items()
                                             if subreddit in ct.index)
                                   },
                                  sort_keys=True) + '\n')
    return pd.DataFrame.from_dict(most_frequent_subs, orient='index'), outf_dims, outf_ct, pd.Series(subreddit_sums)

    # # compute most frequent subs
    # df = pd.DataFrame({k: v for vv in chunk for k, v in vv.items()}).T
    # most_frequent_subs = df.idxmax(axis=1)
    # # compute scores for Waller+Anderson's dimensions
    # normed = df.div(df.sum(axis=1), axis=0)
    # res = normed.apply(lambda row: (row * dims.T).sum(axis=1), axis=1)
    # outf_dims = [json.dumps({n: d.to_dict()}, sort_keys=True) + '\n' for n, d in res.iterrows()]
    # # compute scores for ct
    # res = normed.apply(lambda row: (row * ct.T).sum(axis=1), axis=1)
    # outf_ct = [json.dumps({n: d.to_dict()}, sort_keys=True) + '\n' for n, d in res.iterrows()]
    # # compute users per subreddit
    # df = df[df.fillna(0).astype(bool).sum(
    #     axis=1) > min_subreddits_per_user]  # discard low-freq users from the computation
    # subreddit_sums = df.fillna(0).astype(bool).sum(axis=0)
    # return most_frequent_subs, outf_dims, outf_ct, subreddit_sums


def subreddit_mean_and_variance(chunk, all_subs):
    subreddit_totals = dict()
    users = set()
    for line in chunk:
        user, data = tuple(line.items())[0]
        if user in users: continue
        for subreddit in all_subs:
            subreddit_totals[subreddit] = subreddit_totals.get(subreddit, 0) + data.get(subreddit, 0)
        users.add(user)
    n_users = len(users)
    subreddit_averages = {subreddit: total / n_users for subreddit, total in subreddit_totals.items()}
    subreddit_stds = dict()
    for line in chunk:
        user, data = tuple(line.items())[0]
        if user in users: continue
        for subreddit in all_subs:
            subreddit_stds[subreddit] = subreddit_stds.get(subreddit, 0) + \
                                        (data.get(subreddit, 0)-subreddit_averages[subreddit])**2
    subreddit_stds = {subreddit: np.sqrt(total / n_users) for subreddit, total in subreddit_stds.items()}
    return subreddit_averages, subreddit_stds


def assign_labeler_to_subreddit(external_dir, fpath_histogram_before, out_folder, min_subreddits_per_user=3,
                                min_users_in_subreddit=20):
    # with open(fpath_histogram_before, encoding='utf8') as f:
    #     df = pd.DataFrame({k: v for vv in map(json.loads, f) for k, v in vv.items()}).T
    # df = pd.read_json(fpath_histogram_before, lines=True, orient='index')
    ct = pd.read_csv(os.path.join(external_dir, 'conspiracy_svd_cossim_vector.csv'), index_col=0).rename(
        columns={'similarity': 'conspiracy'})
    dims = pd.read_csv(os.path.join(external_dir, 'scores.csv'), index_col=0)

    most_frequent_subs = list()
    subreddit_sums = list()
    with open(fpath_histogram_before, encoding='utf8') as f, \
            open(os.path.join(out_folder, 'labeler_sub_dimensions.jsonl'), 'w+', encoding='utf8') as outf_dims, \
            open(os.path.join(out_folder, 'labeler_sub_conspiracy.jsonl'), 'w+', encoding='utf8') as outf_ct, \
            Pool(50) as pool:
        for res in pool.imap_unordered(partial(_process_chunk,
                                               ct=ct,
                                               dims=dims,
                                               min_subreddits_per_user=min_subreddits_per_user),
                                       chunkize_iter(map(json.loads, f), 10000)):
            most_frequent_subs_, outf_dims_, outf_ct_, subreddit_sums_ = res
            subreddit_sums.append(subreddit_sums_)
            most_frequent_subs.append(most_frequent_subs_)
            for el in outf_dims_:
                outf_dims.write(el)
            for el in outf_ct_:
                outf_ct.write(el)
            # #compute most frequent subs
            # df = pd.DataFrame({k: v for vv in chunk for k, v in vv.items()}).T
            # most_frequent_subs.append(df.idxmax(axis=1))
            # #compute scores for Waller+Anderson's dimensions
            # normed = df.div(df.sum(axis=1), axis=0)
            # res = normed.apply(lambda row: (row * dims.T).sum(axis=1), axis=1)
            # for n, d in res.iterrows():
            #     outf_dims.write(json.dumps({n: d.to_dict()}, sort_keys=True)+'\n')
            # #compute scores for ct
            # res = normed.apply(lambda row: (row * ct.T).sum(axis=1), axis=1)
            # for n, d in res.iterrows():
            #     outf_ct.write(json.dumps({n: d.to_dict()}, sort_keys=True)+'\n')
            #
            #
            # df = df[df.fillna(0).astype(bool).sum(
            #     axis=1) > min_subreddits_per_user]  # discard low-freq users from the computation
            # subreddit_sums.append(df.fillna(0).astype(bool).sum(axis=0))

    most_frequent_subs = pd.concat(most_frequent_subs)
    most_frequent_subs.to_csv(os.path.join(out_folder, 'labeler_most_frequent_subs.csv'))
    n_users = len(most_frequent_subs)
    del most_frequent_subs

    subreddit_sums = pd.concat(subreddit_sums, axis=1)
    subreddit_sums = subreddit_sums.sum(axis=1)
    remaining_subreddits = list(subreddit_sums[subreddit_sums > min_users_in_subreddit].index)
    print(
        f'{len(remaining_subreddits)} subreddits have over {min_users_in_subreddit} users ({n_users} users {len(subreddit_sums)} subreddits total)')
    del subreddit_sums

    filtered_df = pd.DataFrame(columns=remaining_subreddits, dtype=int)
    with open(fpath_histogram_before, encoding='utf8') as f:
        for chunk in chunkize_iter(map(json.loads, f), 10000):
            df = pd.DataFrame({k: v for vv in chunk for k, v in vv.items()}).T
            df = df[[i for i in remaining_subreddits if i in df.columns]]
            df = df[df.fillna(0).astype(bool).sum(axis=1) > min_subreddits_per_user]
            df = df.div(df.sum(axis=1), axis=0)
            filtered_df = pd.concat((filtered_df, df.fillna(0)))

    # # filtered_df.fillna(0, inplace=True)
    # # most_frequent_subs = df.idxmax(axis=1)
    # # filtered_df = df.dropna(thresh=min_subreddits_per_user, axis=0).dropna(thresh=min_users_in_subreddit,
    # #                                                                        axis=1).fillna(0)
    # # filtered_df = filtered_df.div(filtered_df.sum(axis=1), axis=0)
    # highest_std_subs = filtered_df.apply(zscore).idxmax(axis=1)
    # highest_std_subs.to_csv(os.path.join(out_folder, 'labeler_highest_std_subs.csv'))

    sample = filtered_df
    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(sample, amount_initial_centers).initialize()
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is 20.
    xmeans_instance = xmeans(sample, initial_centers, 20)
    xmeans_instance.process()
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()

    cluster_assocs = dict()
    for cluster_num, cluster in enumerate(clusters):
        cluster_assocs.update(dict(zip(sample.iloc[cluster].index, [cluster_num] * len(cluster))))
    cluster_series = pd.Series(cluster_assocs)[sample.index]
    center_df = pd.DataFrame(centers, columns=sample.columns)
    cluster_series.to_csv(os.path.join(out_folder, 'labeler_clusters.csv'))
    center_df.to_csv(os.path.join(out_folder, 'labeler_cluster_centers.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    interim_dir = os.path.join(project_dir, 'data', 'interim')
    raw_dir = os.path.join(project_dir, 'data', 'raw')
    external_dir = os.path.join(project_dir, 'data', 'external')

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

    # extract_thread_structure(
    #     labeling_fpath=os.path.join(interim_dir, 'labeling_contributions_preprocessed_no_bot.jsonl'),
    #     discussions_fpath=os.path.join(interim_dir, "labeling_discussions_all_filtered_preprocessed_no_bot.jsonl"),
    #     out_fpath=os.path.join(interim_dir, 'thread_structures.jsonl'))
    # filter_threads(in_fpath=os.path.join(interim_dir, 'thread_structures.jsonl'),
    #                seconds_delta=60 * 60,  # 1h
    #                index_delta=25,
    #                min_thread_size=50,
    #                out_folder=os.path.join(interim_dir, 'labeling_discussion_subset'),
    #                )
    ##mv threat_structres.jsonl thread_structures.jsonl
    # consolidate_filtered_threads(discussions_fpath=os.path.join(interim_dir, "labeling_discussions_all_filtered_preprocessed_no_bot.jsonl"),
    #                              filtered_ids_fpath=os.path.join(interim_dir, 'labeling_discussion_subset', 'discussions_by_subthread.jsonl'),
    #                              out_fpath_all=os.path.join(interim_dir, "labeling_subthread_all_filtered_preprocessed_no_bot.jsonl"),
    #                              out_fpath_default=os.path.join(interim_dir, "labeling_subthread_default_filtered_preprocessed_no_bot.jsonl"),
    #                              out_fpath_ct=os.path.join(interim_dir, "labeling_subthread_ct_filtered_preprocessed_no_bot.jsonl"))

    # out_folder = os.path.join(interim_dir, 'counts')
    # compute_baseline_volume(out_folder=out_folder)
    # consolidate_baseline_volume(in_folder=out_folder)

    # labeler_subreddit_distribution(
    #     labeling_fpath=os.path.join(interim_dir, "labeling_contributions_preprocessed_no_bot.jsonl"),
    #     labeler_contributions_fpath=os.path.join(interim_dir, 'labelers_all.jsonl'),
    #     fpath_histogram_before=os.path.join(interim_dir, 'labeler_histograms_before.jsonl'),
    #     fpath_histogram_after=os.path.join(interim_dir, 'labeler_histograms_after.jsonl'), )

    assign_labeler_to_subreddit(external_dir=external_dir,
                                fpath_histogram_before=os.path.join(interim_dir, 'labeler_histograms_before.jsonl'),
                                out_folder=interim_dir,
                                min_subreddits_per_user=3,
                                min_users_in_subreddit=20)
