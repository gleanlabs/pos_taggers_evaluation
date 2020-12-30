# pos_taggers_evaluation

First of all, some figures: around 47.000 in total over the 1007 sentences. Only around 6500 where there are disagreements between the 5 votes (including GT).

- First simple rule: when 4 agree and the GT is different : we take the libraries predictions except for PROPN (GT) /NOUN (libraries) (we PROPN)
===> it's around 2.000 cases

- When 3 agree, there are 2 uniques and the GT is different and agrees with an other library: usually it agrees with spacy (and usually the combinison 
nltk-stanza-flair is right but it's worth reviewing), when it agrees with flair the GT usually always right, when it agrees with nltk the GT usually wrong 
and with stanza usually the GT is true
==> I'll review here


- When 3 agree, there are 3 unique values and the GT is different: we take the libraries predictions except for PROPN/NOUN libke before

- 3 agree 2 uniques same GT: REVIEW

- 3 agree 3 uniques same GT: take GT

- 2 agree 4 unique different GT: take GT (garbbage) 

- 2 agree 3 unique different GT: REVIEW
- 2 agree 4 unique same GT: REVIEW
- 2 agree 3 unique same GT: REVIEW

- 1 agree: few cases - garbbage:  GT
