import re

from spacy.tokens.doc import Doc


class MacIntyreContractions:
    """
    List of contractions adapted from Robert MacIntyre's tokenizer.
    """

    CONTRACTIONS2 = [
        r"(?i)\b(can)(?#X)(not)\b",
        r"(?i)\b(d)(?#X)('ye)\b",
        r"(?i)\b(gim)(?#X)(me)\b",
        r"(?i)\b(gon)(?#X)(na)\b",
        r"(?i)\b(got)(?#X)(ta)\b",
        r"(?i)\b(lem)(?#X)(me)\b",
        r"(?i)\b(mor)(?#X)('n)\b",
        r"(?i)\b(wan)(?#X)(na)\s",
    ]
    CONTRACTIONS3 = [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]
    CONTRACTIONS4 = [r"(?i)\b(whad)(dd)(ya)\b", r"(?i)\b(wha)(t)(cha)\b"]


class RevisedTreeBankWordTokenizerVocab():
    """
    The NLTK tokenizer that has improved upon the TreebankWordTokenizer.

    The tokenizer is "destructive" such that the regexes applied will munge the
    input string to a state beyond re-construction.
    Since this word tokenizer is too agressive for our needs we're implementing
    a revised version
    """

    def __init__(self, vocab):
        self.vocab = vocab


    def __call__(self, text, return_str=False):

        spaces = [True] * len(text)

        return Doc(self.vocab, text, spaces)
