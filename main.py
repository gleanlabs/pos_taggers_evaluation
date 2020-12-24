"""
Script for running pos tagging and comparing results.
"""

import source.tag_pos as tp
import source.tag_statistics as ts
import source.tokenizer_functions as t

# TODO: These are just place fillers. Change to real code.

if __name__ == "__main__":
    t.tokenize()
    tp.pos_tag()
    ts.vote()
    ts.compare()