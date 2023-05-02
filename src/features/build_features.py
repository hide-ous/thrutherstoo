import json
import logging
import os
from itertools import islice
from pathlib import Path

from src.features.preprocess_text import clean_items, preprocess_pre_tokenizing
from src.utils import to_file


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

    if contribution_type in set(['selftext_submission', 'link_submission']):
        item['fullname'] = "t3_" + item.pop('id')
        item['parent_fullname'] = None
        item['link_fullname'] = item['fullname']
    elif contribution_type == 'comment':
        item['fullname'] = "t1_" + item.pop('id')
        item['parent_fullname'] = item.pop('parent_id')
        item['link_fullname'] = item.pop('link_id')
    return item


def preprocess(item, text_field='text', preprocessed_field='preprocessed_text'):
    item[preprocessed_field] = preprocess_pre_tokenizing(item[text_field])
    return item


def stream_normalized_contribution(fpath):
    with open(fpath, encoding='utf8') as f:
        yield from map(normalize_text, map(json.loads, f))


def main():
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

    for fpath in [labeling_fpath,
                  # discussion_fpath, sample_fpath
                  ]:
        out_fpath = os.path.splitext(fpath)[0] + '_preprocessed.jsonl'

        to_file(out_fpath, clean_items(item_stream=
                                       map(preprocess,
                                           stream_normalized_contribution(
                                               fpath)),
                                       text_field='preprocessed_text',
                                       cleaned_text_field='processed_text',
                                       remove_punct=True, remove_digit=True,
                                       remove_stops=True,
                                       remove_pron=False,
                                       lemmatize=True, lowercase=True,
                                       n_process=1
                                       ))


if __name__ == '__main__':
    main()
