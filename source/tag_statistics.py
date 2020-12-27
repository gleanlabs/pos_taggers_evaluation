"""
Analyze tag votes from different packages

"""
from collections import Counter


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def return_majority_token(token_tags:list):
    tags = [i[1] for i in token_tags]
    return most_frequent(token_tags)


def return_number_votes_majority_token(token_tags: list):
    tag = return_majority_token(token_tags)
    return token_tags.count(tag)


def return_unique_tokens(token_tags: list):
    return len(list(set([i[1] for i in token_tags])))


def return_wether_majority_token_equals_gt(token_tags: list):
    tag = return_majority_token(token_tags)
    return tag == token_tags[-1]
