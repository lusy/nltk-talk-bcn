---
title       : NLP 101
author      : lusy (vaseva@mi.fu-berlin.de)
framework   : io2012
highlighter : highlight.js
hitheme     : tomorrow
widgets     : [mathjax]
mode        : selfcontained
---

## Reload Jane Austin

```bash
> switch to console
```

---

## Take a step back

```python
def generate(self, length=100):
    """
    Print random text, generated using a trigram language model.

    :param length: The length of text to generate (default=100)
    :type length: int
    :seealso: NgramModel
    """
    if '_trigram_model' not in self.__dict__:
        print "Building ngram index..."
        estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
        self._trigram_model = NgramModel(3, self, estimator=estimator)
    text = self._trigram_model.generate(length)
    print tokenwrap(text)
```

---

$$
P(W) = P(w_1,w_2,...w_n)
$$

alala
