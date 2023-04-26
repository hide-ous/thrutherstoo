import json
import os
import pathlib

import praw as praw
from prawcore import NotFound
from pmaw import PushshiftAPI
from tqdm import tqdm

from src.utils import chunks, to_file

__r = None


def get_reddit():
    global __r
    if __r is None:
        __r = praw.Reddit()
        __r.read_only = True
    return __r


def yield_content(search_func, prefix, **search_params):
    api_request_generator = search_func(**search_params)
    for content in api_request_generator:
        to_return = content
        to_return['name'] = prefix + to_return['id']
        yield to_return


def scrape_pushshift(search_funcs, prefixes, **search_params):
    for search_func, prefix in zip(search_funcs, prefixes):
        for content in tqdm(yield_content(search_func, prefix, **search_params),
                            'scraping with ' + search_func.__name__):
            yield content


def rehydrate_content_pushshift(ids):
    comment_ids = list(filter(lambda x: x.startswith('t1_'), ids))
    submission_ids = list(filter(lambda x: x.startswith('t3_'), ids))
    api = PushshiftAPI()
    # need to chunkize because of:
    # NotImplementedError: When searching by ID, number of IDs must be
    # fewer than the max number of objects in a single request (1000).
    for chunk in chunks(submission_ids, 1000):
        for submission in yield_content(api.search_submissions, 't3_', ids=chunk):
            yield submission
    for chunk in chunks(comment_ids, 1000):
        for comment in yield_content(api.search_comments, 't1_', ids=chunk):
            yield comment

def search_content_pushshift(**args):
    api = PushshiftAPI()
    # need to chunkize because of:
    for submission in yield_content(api.search_submissions, 't3_', **args):
        yield submission
    for comment in yield_content(api.search_comments, 't1_', **args):
        yield comment


def search_pushshift(store_path, **args):
    with open(store_path, 'w+') as f:
        for content in tqdm(search_content_pushshift(**args), "searching pushshift"):
            f.write(json.dumps(content) + '\n')


def rehydrate_parents_pushshift(things):
    parent_ids = set()
    for thing in things:
        if 'parent_id' in thing:
            parent_ids.add(thing['parent_id'])
    for parent in tqdm(rehydrate_content_pushshift(parent_ids),
                       "rehydrating parents",
                       len(parent_ids)):
        yield parent


def rehydrate_content_praw(ids):
    r = get_reddit()
    for thing in r.info(ids):
        try:
            data = thing._fetch_data()
            if isinstance(data, list):
                data = data[0]
            for child in data['data']['children']:
                yield child['data']
        except Exception as e:
            print(e)
            yield None  # in case something does not exist


def rescrape_praw(store_path, ids):
    with open(store_path, 'w+') as f:
        for content in tqdm(rehydrate_content_praw(ids), "rescraping with praw", len(ids)):
            f.write(json.dumps(content) + '\n')


def get_user_pushshift(user):
    api = PushshiftAPI()
    search_funcs = [api.search_submissions, api.search_comments]
    prefixes = ['t3_', 't1_']
    search_params = dict(author=user, )
    yield from scrape_pushshift(
        search_funcs=search_funcs,
        prefixes=prefixes,
        **search_params
    )


def get_user_praw(user):
    # maxes out at 1000 comments
    r = get_reddit()
    user_stream = r.redditor(user)
    for thing in user_stream.new(limit=None):  # could also use hot, controversial, top
        data = thing._fetch_data()
        if isinstance(data, list):
            data = data[0]
        for child in data['data']['children']:
            yield child['data']


def get_user_info_praw(name):
    r = get_reddit()
    user = r.redditor(name)
    return user._fetch_data()['data']

def get_users_info_praw(unique_users):
    r = get_reddit()

    for user in unique_users:
        try:
            user_=r.redditor(user)
            yield user_._fetch_data()['data']
        except NotFound as e:
            yield {"display_name":"u_"+user, "_comment": "not found:" + repr(e)}
if __name__ == '__main__':

    import pickle
    with open(os.path.join(os.path.dirname('.'), os.pardir, os.pardir, 'data', 'interim', 'subreddit_users.csv'),
              'rb') as f:
        unique_users = pickle.load(f)
    out_path = os.path.join(os.path.dirname('.'), os.pardir, os.pardir, 'data', 'interim', 'subreddit_users_info.njson')
    to_file(out_path, tqdm(get_users_info_praw(unique_users), desc='fetching user information', total=len(unique_users)))
    # print(get_user_info_praw('hide_ous'))
    # from src.data.load_income import CensusMSAIncome
    # from src.data.load import load_embedding
    #
    # vectors, meta = load_embedding()
    # census = CensusMSAIncome(vectors)
    # census_data = census.data()['uscensus_income']
    # subreddits = census_data.index.tolist()
    # # print(subreddits)
    #
    # dump_dir = os.path.join(os.path.dirname('.'), os.pardir, os.pardir, 'data', 'raw', 'subreddit_data')
    # pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
    # for subreddit in subreddits:
    #
    #     store_path = os.path.join(dump_dir, "{}.pushshift.njson".format(subreddit))
    #     api = PushshiftAPI()
    #     search_params = dict(subreddit=subreddit)
    #     if not os.path.exists(store_path):
    #         print('scraping', subreddit)
    #         to_file(store_path=store_path,
    #                 stream=scrape_pushshift(
    #                          search_funcs=[api.search_submissions, api.search_comments],
    #                          prefixes=['t3_', 't1_'],
    #                          **search_params
    #                          ))
    #
    # # to_file('hide_ous.pushshift.njson', get_user_pushshift('hide_ous', ))
    # # to_file('hide_ous.praw.njson', get_user_praw('hide_ous', ))
