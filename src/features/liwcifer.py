"""Main module."""
import re
import json
from functools import partial

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from collections import defaultdict

def read_liwc(lexicon_path='LIWC2015.jsonl'):
    with open(lexicon_path, 'r') as f:
        return json.load(f)

def create_matchings(vocabulary_words, lexica):
    #cv = CountVectorizer(input='file', max_df=.5, lowercase=True)
    #data_sample = open(comments_path)
    # vocabulary_words = vectorize_comments(df_comments)[1]
    # X = cv.fit(data_sample)
    # lexica = read_liwc(lexicon_path)
    # all_terms = set()
    # for lexicon in lexica:
    #    all_terms.update(lexica[lexicon])

    mapping = {}
    inverse_mapping = {}
    for word in vocabulary_words:
        tmp_list = []
        for lexicon_name in lexica:
            if re.match(lex_to_regex(lexica[lexicon_name]), word) is not None:
                tmp_list.append(lexicon_name)
        mapping[word] = tmp_list

    for lexicon_name in lexica:
        tmp_inv = []
        for vocab_word in mapping:
            if lexicon_name in mapping[vocab_word]:
                tmp_inv.append(vocab_word)
        inverse_mapping[lexicon_name] = tmp_inv

    return mapping, inverse_mapping

def vectorize_comments(dataframe, content_field="title"):
    # returns document term matrix and feature names of data sample in file in path
    cv = CountVectorizer(max_df=.5, lowercase=True)
    data_sample = list(dataframe[content_field])

    ids = list(dataframe.index.values)

    X = cv.fit_transform(data_sample)
    return ids, cv.get_feature_names_out(), X

def lex_to_regex(lexicon_list):

    regex = ''
    for term in lexicon_list:
        term = term.replace("*",".*")
        if '(discrep' in term:
            term = term.replace('(discrep)', '(?P=Discrep)')
        elif ('(' in term) and (')' in term):
            term = term.replace('(', '(?:')

        else:
            # pass
            term = term.replace(")",r"[)]")
            term = term.replace("(", r"[(]")
        regex = regex + r'|\b'+ term + r'\b'
    regex = regex.removeprefix('|')
    raw_s = r'{}'.format(regex)
    return raw_s

def extract_liwcs(document_df, lexica, content_field='text'):
    ids, feature_names, X = vectorize_comments(document_df, content_field=content_field)
    print(ids)
    print('feature names:')
    print(feature_names)
    print(len(feature_names))
    print('printing X array:')
    print(X.toarray())
    # cv = CountVectorizer(max_df=.5, lowercase=True)
    # data_sample = list(df_index_body["title"])
    # ids = list(df_index_body.index.values)
    # X = cv.fit_transform(data_sample)
    df = pd.DataFrame(data=X.A, index=ids, columns=feature_names)
    print(df)

    mapping, inv_mapping = create_matchings(feature_names, lexica)
    # to test 4.5:
    # print('First mapping:')
    # print(mapping)
    # print('Inverse mapping:')
    # print(inv_mapping)
    # task 4.6

    lex_sums = {}
    for lexicon in inv_mapping:
        lexicon_match_indices = df.columns.get_indexer(inv_mapping[lexicon]).tolist()
        lex_sums[lexicon] = df.iloc[:, lexicon_match_indices].sum(axis=1)
    df_lex = pd.DataFrame.from_dict(lex_sums)
    print(df_lex)

    # task 4.7 - normalize matrix
    dividend_array = df_lex.to_numpy()
    # added the number of words in column 'title' to the dataframe
    # df_index_body['len_title'] = df_index_body['title'].str.split().str.len()
    # divisor_list = df_index_body['len_title']
    # divisor_list = df_index_body.len_title.values.tolist()
    divisor_list = X.sum(axis=1).getA1()[:,np.newaxis]#.tolist()

    norm_array = np.divide(dividend_array, divisor_list, out=np.zeros_like(dividend_array), where=divisor_list!=0)
    print(divisor_list)
    print('NORMALIZED np array:')
    print(norm_array)
    return norm_array
    # test if sum of row is always 1
    # row_sums1 = norm_array.sum(axis=1).tolist()
    # print('test - row sums')
    # print(row_sums1)

def get_matchers(lexica):
    regexes_dict = dict()
    for lexicon_name, lexicon_list in sorted(lexica.items()):
        the_regex= lex_to_regex(lexicon_list)
        regexes_dict[lexicon_name] = the_regex
    regexes_dict['Posemo']=regexes_dict['Posemo'].replace('?P=Discrep', regexes_dict['Discrep'])
    #TODO: there is a (53) group before a "like"; investigate what it means. is that the index of a LIWC category?
    regexes =list()
    for lexicon_name, lexicon_re in sorted(regexes_dict.items()):
        the_regex= r'(?P<{}>{})'.format(lexicon_name, lexicon_re)
        regexes.append(the_regex)
    regexes.append(r'(?P<Tokens>\b\w+\b)')
    return [re.compile(regex, flags=re.I|re.M|re.U|re.DOTALL) for regex in regexes]
    # return re.compile(r'|'.join(regexes), flags=re.I|re.M|re.U|re.DOTALL)

def match_sent(sent: str, matchers):
    """
    returns a dictionary where the key is the name of the lexicon, and the value
    a list of the matching strings
    """
    to_return=defaultdict(list)
    for matcher in matchers:
        for match in re.finditer(matcher, sent):
            for lex_name, matched_word in match.groupdict().items():
                if matched_word is not None:
                    to_return[lex_name].append(matched_word)
    return dict(to_return)

def bag_of_lexicons(sent:str, matcher):
    return pd.Series({k: len(v) for k, v in match_sent(sent, matcher).items()})

def df_liwcifer(df:pd.DataFrame, text_col, matcher):
    return df[text_col].apply(partial(bag_of_lexicons, matcher=matcher)).fillna(0)

if __name__ == '__main__':
    lexicon_path = '../LIWC2015.jsonl'
    lexica = read_liwc(lexicon_path)
    matchers = get_matchers(lexica)
    # for sent in ['this is a document',
    #                                     'this is another document',
    #                                     'there are so many documents in here']:
    #     print(sent, match_sent(sent, matchers))
    document_df = pd.DataFrame({'text':['this is a document',
                                        'this is another document',
                                        'there are so many documents in here']},
                               index=['a', 'b', 'c'])
    print(df_liwcifer(document_df,'text', matchers))
    # extract_liwcs(document_df, lexica, content_field='text')
