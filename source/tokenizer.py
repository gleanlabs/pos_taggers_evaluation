"""
Module for tokenize text prior to pos tagging
"""

import nltk


def tokenize(text: str):
    return text.split(' ')

