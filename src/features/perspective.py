#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from googleapiclient import discovery
import time
from ratelimit import limits, sleep_and_retry

REQUESTED_ATTRIBUTES_ALL = {"TOXICITY": {}, "SEVERE_TOXICITY": {}, "IDENTITY_ATTACK": {}, "INSULT": {}, "PROFANITY": {},
                            "SEXUALLY_EXPLICIT": {}, "THREAT": {}, "FLIRTATION": {}, "ATTACK_ON_AUTHOR": {},
                            "ATTACK_ON_COMMENTER": {},
                            "INCOHERENT": {}, "INFLAMMATORY": {}, "LIKELY_TO_REJECT": {}, "OBSCENE": {}, "SPAM": {},
                            "UNSUBSTANTIAL": {}, }
REQUESTED_ATTRIBUTES_TOXICITY = {"TOXICITY": {}}

ONE_SECOND = 1
ONE_MINUTE = 60 * ONE_SECOND


@sleep_and_retry
@limits(calls=3000, period=ONE_MINUTE)
def get_toxicity_score(comment, service, requested_attributes=REQUESTED_ATTRIBUTES_TOXICITY, languages=['en']):
    """given a comment, return its scores via perspective API
    
    params:
    comment: unicode string
    service: commentanalyzer instance
    
    returns: dict, see perspective doc
    """
    analyze_request = {
        'comment': {'text': truncate_str(comment)},
        'languages': languages,
        'requestedAttributes': requested_attributes
    }
    response = service.comments().analyze(body=analyze_request).execute()
    return response


def parse_summary_scores(response):
    """given a response from the perspective API, return the summary score for each metric present"""
    scores = response['attributeScores']
    return {score_name: scores[score_name]['summaryScore']['value'] for score_name in scores}


def truncate_str(s, n_bytes=20480):
    return s.encode('utf-8')[:n_bytes].decode('utf-8', 'ignore')


def get_toxicity_scores(comments, perspective_key, max_retries=3, requested_attributes=REQUESTED_ATTRIBUTES_TOXICITY,
                        languages=['en']):
    """given a list of comments, return their scores through the API
    
    params:
    comments: list of unicode strings
    authors: string the api key in plain text
    max_retries: number of times to repeat the request if unsuccessful, e.g. becasue of quota limits
    
    returns:
    list of scores (dict as per API), -1 if perspective isn't able to parse the sentence, np.nan if service error
    """
    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=perspective_key,
                              discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                              static_discovery=False, )
    scores = list()
    for comment in comments:
        retries = 0
        done = False
        score = np.nan
        while (not done) and (retries < max_retries):
            try:
                score = get_toxicity_score(comment, service, requested_attributes, languages)
                done = True
            except Exception as e:
                if e.resp['status'] == '400':
                    score = -1
                    done = True
                    print(e)
                else:
                    print(e)
                    retries += 1
                    time.sleep(60)
        scores.append(score)
        time.sleep(1)
    return scores


if __name__ == '__main__':
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    key = os.environ['PERSPECTIVE_KEY']
    comments = [u'you ugly duck', u'the sun shines today', u'steven wilson is the god of prog']
    print(list(zip(comments, map(parse_summary_scores,
                                 get_toxicity_scores(comments, key, requested_attributes=REQUESTED_ATTRIBUTES_ALL)))))
