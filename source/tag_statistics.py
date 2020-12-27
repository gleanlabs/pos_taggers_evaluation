"""
Analyze tag votes from different packages

"""

def map_results_to_universal_tags(raw_token: tuple):
    mapping = _read_tag_map()
    if source == 'nltk':
        dict_mapping = mapping['PTB-UNIV']
        results_mapped = [(tags[0], dict_mapping[tags[1]]) for tags in raw_tokens]
    if source == 'stanza':
        dict_mapping = mapping['PTB-UNIV']
        results_mapped = [(tags[0], dict_mapping[tags[1]]) for tags in raw_tokens]
    if source == 'spacy':
        dict_mapping = mapping['UNIV']
        results_mapped = [(tags[0], dict_mapping[tags[1]]) for tags in raw_tokens]
    if source == 'flair':
        dict_mapping = mapping['PTB-UNIV']
        results_mapped = [(tags[0], dict_mapping[tags[1]]) for tags in raw_tokens]
    return results_mapped


if (index in spacy_index) & (
        index in stanza_index) & (
        index in gc_index) & (
        index in nltk_index) & (
        index in flair_index):
    preds = [gt, flair[flair_index.index(index)], gc[gc_index.index(index)], nltk[nltk_index.index(index)],
             stanza[stanza_index.index(index)],
             spacy[spacy_index.index(index)]]
    most_frequent_val = most_frequent(preds)
    list_tokens.append(
        [tok, gt, most_frequent_val, preds.count(most_frequent_val),
         len(list(set(preds))),
         most_frequent_val == gt,
         preds])
df_pos.loc[i, 'gt_new'] = str(list_tokens)