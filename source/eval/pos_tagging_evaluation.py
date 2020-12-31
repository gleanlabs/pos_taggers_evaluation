import spacy
from source.tag_pos import _read_tag_map, map_results_to_universal_tags, _pos_tag_sentence
from source.pos_taggers_functions import split_labels_articles_that_need_to, _split_composite_pos_tokens
import nltk

spacy.load('en_core_web_sm')
import numpy as np
import ast
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def evaluation(file):
    import pandas as pd

    df = pd.read_csv(file)
    df['GT'] = df['sentence'].apply(
        lambda x: [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('article', x), 'article')])
    print(df['GT'])

    # nltk
    df['nltk'] = df['sentence'].apply(
        lambda x: [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('nltk', x), 'nltk')])
    print('nltk')

    # we remove sentences when things could be removed when manually reviewing
    df['same'] = df[['GT', 'nltk']].apply(lambda x: 1 if len(x[0]) == len(x[1]) else 0, axis=1)
    df = df[df.same == 1]

    # stanza
    df['stanza'] = df['sentence'].apply(
        lambda x: [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('stanza', x), 'stanza')])
    print('stanza')
    df['same'] = df[['GT', 'stanza']].apply(lambda x: 1 if len(x[0]) == len(x[1]) else 0, axis=1)
    df = df[df.same == 1]

    # spacy
    df['spacy'] = df['sentence'].apply(
        lambda x: [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('spacy', x), 'spacy')])
    print('spacy')

    df['same'] = df[['GT', 'spacy']].apply(lambda x: 1 if len(x[0]) == len(x[1]) else 0, axis=1)
    df = df[df.same == 1]

    df.to_csv('test_set_pos_tagging.csv')

    # Take only Basel's corrected pos
    df = pd.read_csv('test_set_pos_tagging.csv')[500:]

    df['nltk'] = df['nltk'].apply(lambda x: ast.literal_eval(x))
    df['spacy'] = df['spacy'].apply(lambda x: ast.literal_eval(x))
    df['stanza'] = df['stanza'].apply(lambda x: ast.literal_eval(x))
    df['GT'] = df['GT'].apply(lambda x: ast.literal_eval(x))

    # we check whenever the 3 libraries agree to get the confusion matrices + classification reports only for the tokens where
    # there are disagreements
    df['agree'] = df[['nltk', 'spacy', 'stanza']].apply(
        lambda x: [1 if nl == sp == st else 0 for nl, sp, st in zip(x[0], x[1], x[2])], axis=1)

    flat_list_nltk = [item for sublist in
                      df[['nltk', 'agree']].apply(lambda x: [nltk for nltk, agree in zip(x[0], x[1]) if agree == 0],
                                                  axis=1).tolist() for
                      item in sublist]
    flat_list_spacy = [item for sublist in
                       df[['spacy', 'agree']].apply(lambda x: [nltk for nltk, agree in zip(x[0], x[1]) if agree == 0],
                                                    axis=1).tolist() for
                       item in sublist]
    flat_list_stanza = [item for sublist in
                        df[['stanza', 'agree']].apply(lambda x: [nltk for nltk, agree in zip(x[0], x[1]) if agree == 0],
                                                      axis=1).tolist() for
                        item in sublist]
    flat_list_gt = [item for sublist in
                    df[['GT', 'agree']].apply(lambda x: [nltk for nltk, agree in zip(x[0], x[1]) if agree == 0],
                                              axis=1).tolist() for item in
                    sublist]

    array_nltk = confusion_matrix(flat_list_gt, flat_list_nltk, labels=list(set(flat_list_gt)))
    array_spacy = confusion_matrix(flat_list_gt, flat_list_spacy, labels=list(set(flat_list_gt)))
    array_stanza = confusion_matrix(flat_list_gt, flat_list_stanza, labels=list(set(flat_list_gt)))

    # nltk confusion matrix + classification report
    df_cm = pd.DataFrame(array_nltk, index=[i for i in list(set(flat_list_gt))],
                         columns=[i for i in list(set(flat_list_gt))])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.title('nltk confusion matrix_' + str(np.sum(array_nltk)))
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.show()
    print("NLTK:")
    print(classification_report(flat_list_gt, flat_list_nltk, labels=list(set(flat_list_gt))))

    # spacy confusion matrix + classification report
    df_cm = pd.DataFrame(array_spacy, index=[i for i in list(set(flat_list_gt))],
                         columns=[i for i in list(set(flat_list_gt))])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.title('spacy confusion matrix_' + str(np.sum(array_spacy)))
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.show()
    print("SPACY:")
    print(classification_report(flat_list_gt, flat_list_spacy, labels=list(set(flat_list_gt))))

    # stanza confusion matrix + classification report
    df_cm = pd.DataFrame(array_stanza, index=[i for i in list(set(flat_list_gt))],
                         columns=[i for i in list(set(flat_list_gt))])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.title('stanza confusion matrix_' + str(np.sum(array_stanza)))
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.show()
    print("STANZA:")
    print(classification_report(flat_list_gt, flat_list_stanza, labels=list(set(flat_list_gt))))
