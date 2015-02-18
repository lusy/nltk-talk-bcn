---
title       : NLP 101
subtitle    : aided by python and nltk
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

## Ngram?

> * a sequence of n consecutive words (tokens)
> * *I like fluffy ponies.*
  * bigrams: "I like", "like fluffy", "fluffy ponies"
  * trigrams: "I like fluffy", "like fluffy ponies"

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
$$ P(\text{"I like fluffy ponies"}) = P(\text{I}) \times P(\text{like}|\text{I}) \times P(\text{fluffy}|\text{I like}) \times P(\text{ponies}|\text{I like fluffy}) $$

---

## How do we compute the probabilities?

### Just count occurancies?

$$
\begin{align*}
P(\text{ponies}|\text{I like fluffy}) = \\
\\
\frac{\text{count}(\text{I like fluffy ponies})}{\text{count}(\text{I like fluffy})}
\end{align*}
\\
$$

---

## Markov assupmtion: simplify!

> the probability for the upcoming word given the entire context would be similar to the probability for it given just the last couple of words

$$
P(\text{rainbows}|\text{I like pink fluffy ponies dancing on}) \approx P(\text{rainbows}|\text{on})
$$

or

$$
P(\text{rainbows}|\text{I like pink fluffy ponies dancing on}) \approx P(\text{rainbows}|\text{dancing on})
$$

---

## Simplest case: Unigram model

$$
P(w_1,w_2,\dotsc,w_n) \approx P(w_1)P(w_2) \dotsm P(w_n)
$$

in other words:

$$
P(\text{rainbows}|\text{I like pink fluffy ponies dancing on}) \approx P(\text{rainbows})
$$

---

## Ngram models

### bigram model

conditions on the previous word:

$$ P_{MLE}(w_i|w_{i-1}) = \frac{\text{count}(w_{i-1},w_i)}{\text{count}(w_{i-1})}$$

---

## Ngram models

### trigram model

conditions on the sequence of the previous 2 words:

$$
P(w_i|w_{i-2},w_{i-1}) = \frac{\text{count}(w_{i-2},w_{i-1},w_i)}{\text{count}(w_{i-2},w_{i-1})}
$$

$$
\begin{multline}
P(\text{*s* I like pink fluffy unicorns *e*}) = \\
P(\text{I}|\text{*s*})P(\text{like}|\text{*s* I})P(\text{pink}|\text{I like})P(\text{fluffy}|\text{like pink})\\
P(\text{unicorns}|\text{pink fluffy})P(\text{*e*}|\text{fluffy unicorns})
\end{multline}
$$


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

$ P(\text{dinosaurs}|\text{pink fluffy}) = 0 $ (doesn't appear in the training set)

$\rightarrow$ so we have no chance to predict it

---

## Generalization: Add 1/Laplace smoothing

### Intuition: add 1 to all the counts in the training set

|          | I   | like   | pink   | fluffy   | unicorns   |
|----------|:---:|:------:|:------:|:--------:|:----------:|
| I        | 0   | 23     | 0      | 0        | 0          |
| like     | 25  | 4      | 35     | 20       | 29         |
| pink     | 0   | 8      | 0      | 5        | 3          |
| fluffy   | 0   | 2      | 2      | 0        | 6          |
| unicorns | 3   | 1      | 0      | 0        | 0          |

---

## Generalization: Add 1/Laplace smoothing

$$
P_{add1} (w_i|w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + 1}{\text{count}(w_{i-1}) + V}
$$

$$
(V = \text{number of occurancies of } w_{i-1} \text{ in the training set} )
$$

* Not really used with ngram models.
* Changes probabilities massively!

---

## Generalization: Backoff

### Intuition:
use less context for unknown stuff.

So if we have good evidence we use trigrams, otherwise bigrams, otherwise unigrams.

---

## Taking another look...

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

## So what's next?

```python
>>> import nltk
>>> nltk.chat.chatbots()
```

```
Which chatbot would you like to talk to?
1: Eliza (psycho-babble)
2: Iesha (teen anime junky)
3: Rude (abusive bot)
4: Suntsu (Chinese sayings)
5: Zen (gems of wisdom)

Enter a number in the range 1-5: 1
========================================================================
Hello.  How are you feeling today?
>
```

It's your turn ;)

---

## Thanks!
### a.k.a. references

* [NLTK Book](http://www.nltk.org/book/)
* Stanford Coursera class on NLP: Dan Jurafsky + Christopher Manning
* ... and NLTK source code ;)
* [slidify](http://slidify.org/) for building these slides
