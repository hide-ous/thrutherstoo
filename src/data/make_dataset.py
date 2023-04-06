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

CONSPIRACY_THEORIST_RE = '(conspiracist)|(conspiracy theorist)'


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    with open(output_filepath, 'w+', encoding='utf8') as f:
        for contribution in read_zst(input_filepath):
            text = contribution['body']
            if not text: continue
            if re.findall(CONSPIRACY_THEORIST_RE, text, flags=re.I | re.DOTALL | re.U | re.M):
                f.write(json.dumps(contribution)+'\n')


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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    infile = os.path.join(project_dir, 'data', 'external', 'RC_2009-09.zst')
    outfile = os.path.join(project_dir, 'data', 'interim', 'RC_2009-09.jsonl')
    main(input_filepath=infile, output_filepath=outfile)
