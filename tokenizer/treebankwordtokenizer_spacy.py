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

        # Starting quotes.
        STARTING_QUOTES = [
            (re.compile(u"([«“‘„]|[`]+)", re.U), r" \1 "),
            (re.compile(r"^\""), r"``"),
            (re.compile(r"(``)"), r" \1 "),
            (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
            (re.compile(r"(?i)(\')(?!re|ve|ll|m|t|s|d)(\w)\b", re.U), r"\1 \2"),
        ]

        # Ending quotes.
        ENDING_QUOTES = [
            (re.compile(u"([»”’])", re.U), r" \1 "),
            (re.compile(r'"'), " '' "),
            (re.compile(r"(\S)(\'\')"), r"\1 \2 "),
            (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
            (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
        ]

        # For improvements for starting/closing quotes from TreebankWordTokenizer,
        # see discussion on https://github.com/nltk/nltk/pull/1437
        # Adding to TreebankWordTokenizer, nltk.word_tokenize now splits on
        # - chervon quotes u'\xab' and u'\xbb' .
        # - unicode quotes u'\u2018', u'\u2019', u'\u201c' and u'\u201d'
        # See https://github.com/nltk/nltk/issues/1995#issuecomment-376741608
        # Also, behavior of splitting on clitics now follows Stanford CoreNLP
        # - clitics covered (?!re|ve|ll|m|t|s|d)(\w)\b

        # Punctuation.
        PUNCTUATION = [
            (re.compile(r'([^\.])(\.)([\]\)}>"\'' u"»”’ " r"]*)\s*$",
                        re.U), r"\1 \2 \3 "),
            (re.compile(r"([:,])([^\d])"), r" \1 \2"),
            (re.compile(r"([:,])$"), r" \1 "),
            # See https://github.com/nltk/nltk/pull/2322
            (re.compile(r"\.{2,}", re.U), r" \g<0> "),
            # GG when writing about code - a lot of these are included
            # and not should not be used as punctuation directives
            # (re.compile(r"[;@#$%&]"), r" \g<0> "),
            (
                re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
                r"\1 \2\3 ",
            ),  # Handles the final period.
            (re.compile(r"[?!]"), r" \g<0> "),
            (re.compile(r"([^'])' "), r"\1 ' "),
            # See https://github.com/nltk/nltk/pull/2322
            (re.compile(r"[*]", re.U), r" \g<0> "),
        ]

        # Pads parentheses
        PARENS_BRACKETS = (re.compile(r"[\(\)]"), r" \g<0> ")

        DOUBLE_DASHES = (re.compile(r"--"), r" -- ")

        # List of contractions adapted from Robert MacIntyre's tokenizer.
        _contractions = MacIntyreContractions()
        CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
        CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

        for regexp, substitution in STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Handles parentheses.
        regexp, substitution = PARENS_BRACKETS
        text = regexp.sub(substitution, text)

        # Handles double dash.
        regexp, substitution = DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        # add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in CONTRACTIONS2:
            text = regexp.sub(r" \1 \2 ", text)
        for regexp in CONTRACTIONS3:
            text = regexp.sub(r" \1 \2 ", text)

        # We are not using CONTRACTIONS4 since
        # they are also commented out in the SED scripts
        # for regexp in self._contractions.CONTRACTIONS4:
        #     text = regexp.sub(r' \1 \2 \3 ', text)

        spaces = [True] * len(text.split())

        return Doc(self.vocab, text, spaces) if return_str else Doc(self.vocab, text.split(), spaces)
