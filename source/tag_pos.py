"""
Module which tags POS given a list of sentences and a package
"""

from source.tokenizer import tokenize


def _choose_package(package_name: str):
    return ""


def _pos_tag_sentence(package_name: str, sent: str):
    pos_tagger = _choose_package(package_name)
    res = []
    for s in sent:
        res.append(pos_tagger.tag(tokenize(sent)))
    return res


def _pos_tag_batch(package_name: str, batch: list[list[str]]): # check syntax for type hints
    pos_tagger = _choose_package(package_name)
    res = []
    for b in batch:
        for sent in b:
            res.append(pos_tagger.tag(tokenize(sent)))
    return res

def _read_tag_map(tag_path: str):
    tag_map = ""
    return tag_map

def tag_pos(package_name: str, sentences: list):
    if package_name in ['stanza', 'flair']:
        result = [_pos_tag_batch(batch) for batch in sentences]
        return [r for r in [res for res in result]]  # flatten list
    else:
        return [_pos_tag_sentence(sent) for sent in sentences]

def map_results_to_universal_tags(raw_tokens: list[tuple], tag_map):
    results_mapped = []
    return results_mapped