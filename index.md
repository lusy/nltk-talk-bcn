---
title       : NLP 101
subtitle    : working title
author      : lusy (vaseva@mi.fu-berlin.de)
framework   : io2012
highlighter : highlight.js
hitheme     : tomorrow
widgets     : [mathjax]
mode        : selfcontained

---&userdefined

*** =center

[https://lusy.github.io/nltk-talk-bcn/](https://lusy.github.io/nltk-talk-bcn/)


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

##  What is a (probabilistic) language model?

### Model that computes the probabilities for:

* a sentence to appear $P(W)$
* an upcoming word of a sentence $P(w_n|w_1,w_2,...,w_{n-1})$

---

## Some maths

### chain probability rule

$$ P(A|B) = \frac{P(A,B)}{P(B)} \Rightarrow  P(A,B) = P(A|B) \times P(B) $$

generalize:
$$ P(x_1,x_2,x_3,\dotsc,x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2) \dotsm P(x_n|x_1,...,x_{n-1}) $$

so:
$$ P(\text{"I like ice cream"}) = P(\text{I}) \times P(\text{like}|\text{I}) \times P(\text{ice}|\text{I like}) \times P(\text{cream}|\text{I like ice}) $$

---

## How do we compute the probabilities?

### Just count occurancies?

$$
\begin{align*}
P(\text{sauce}|\text{I like strawberry ice cream with caramel}) = \\
\\
\frac{\text{count}(\text{I like strawberry ice cream with caramel sauce})}{\text{count}(\text{I like strawberry ice cream with caramel})}
\end{align*}
\\
$$

---

## Markov assupmtion: simplify!

> the probability for the upcoming word given the entire context would be similar to the probability for it given just the last couple of words

$$
P(\text{sauce}|\text{I like strawberry ice cream with caramel}) \approx P(\text{sauce}|\text{caramel})
$$

or

$$
P(\text{sauce}|\text{I like strawberry ice cream with caramel}) \approx P(\text{sauce}|\text{with caramel})
$$

---

## Simplest case: Unigram model

$$
P(w_1,w_2,\dotsc,w_n) \approx P(w_1)P(w_2) \dotsm P(w_n)
$$

in other words:

$$
P(\text{sauce}|\text{I like strawberry ice cream with caramel}) \approx P(\text{sauce})
$$

---

## Ngram models

### bigram model

conditions on the previous word:

$$ P_{MLE}(w_i|w_{i-1}) = \frac{\text{count}(w_{i-1},w_i)}{\text{count}(w_{i-1})}$$

$$
\begin{multline}
P(\text{*s* I like strawberry ice cream with caramel sauce *e*}) = \\
P(\text{I}|\text{*s*})P(\text{like}|\text{I})P(\text{strawberry}|\text{like})P(\text{ice}|\text{strawberry})P(\text{cream}|\text{ice}) \\
P(\text{with}|\text{cream})P(\text{caramel}|\text{with})P(\text{sauce}|\text{caramel})P(\text{*e*}|\text{sauce})
\end{multline}
$$

---

## Ngram models

### trigram model

conditions on the sequence of the previous 2 words:

$$
P(w_i|w_{i-2},w_{i-1}) = \frac{\text{count}(w_{i-2},w_{i-1},w_i)}{\text{count}(w_{i-2},w_{i-1})}
$$

$\rightarrow$ n-gram models are imperfect modellings of language (language has more complicated long-distance dependencies)

   But they often are good enough for the computations we are interested in (e.g. generate text in the same style as text X).

---

## Language Models

* So we train the chosen model on a training set.
* However, if we use the model in its "pure" form, no unseen ngrams could be generated (predicted)!
* And the best language model is one that best predicts a (unseen) test set.

$\rightarrow$ generalization: that's what we need the estimator for (and for smoothing)

```python
def generate(self, length=100):
    if '_trigram_model' not in self.__dict__:
        print "Building ngram index..."
        estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
        self._trigram_model = NgramModel(3, self, estimator=estimator)
    text = self._trigram_model.generate(length)
    print tokenwrap(text)
```

---

## Generalization

### Intuition

$ P(\text{topping}|\text{with caramel}) = 0 $ (doesn't appear in the training set)

$\rightarrow$ so we have no chance to predict it

---

## Generalization: Add 1/Laplace smoothing

### Intuition:
When we count occurencies in the training set, we add 1 to all the counts;
that way we have a small probability set apart for "others", e.g. for unseen stuff.

$$
P_{add1} (w_i|w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + 1}{\text{count}(w_{i-1}) + V}
$$

$$
(V = \text{number of occurancies of } w_{i-1} \text{ in the training set} )
$$

* Not really used with ngram models.
* Changes probabilities massively! (when we recompute them according to these new counts)
* Used in domains, where number of zeroes which need to be smoothed isn't so enormous.

---

## Generalization: Backoff

used by the NgramModel in nltk

```python
def prob(self, word, context):
    """
    Evaluate the probability of this word in this context using Katz Backoff.

    :param word: the word to get the probability of
    :type word: str
    :param context: the context the word is in
    :type context: list(str)
    """

    context = tuple(context)
    if (context + (word,) in self._ngrams) or (self._n == 1):
        return self[context].prob(word)
    else:
        return self._alpha(context) * self._backoff.prob(word, context[1:])
```

---

## Generalization: Backoff

### Intuition:
use less context for unknown stuff

So if we have good evidence we use trigrams, otherwise bigrams, otherwise unigrams

---

## Thanks!
### a.k.a. references

* [NLTK Book](http://www.nltk.org/book/)
* Stanford Coursera class on NLP: Dan Jurafsky + Christopher Manning
* ... and NLTK source code ;)
* [slidify](http://slidify.org/) for building these slides
