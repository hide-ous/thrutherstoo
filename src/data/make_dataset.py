# -*- coding: utf-8 -*-
import json
import os.path
import re
from json import JSONDecodeError

import click
import logging
from pathlib import Path
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


def parse_files():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
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

    for infile in contribution_fpaths:
        outfile = infile[:-len('.zst')] + '_labeling.zst'
        text_field = "selftext" if "RS" in infile else "body"
        main(input_filepath=infile, output_filepath=outfile,
             text_field=text_field)


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

    outfile = os.path.join(project_dir, 'data', 'interim',
                           'labeling_contributions.jsonl')
    search_pushshift(store_path=outfile,
                     q='''conspiracist|conspiracists|"conspiracy theorist"|"conspiracy theorists"''')
