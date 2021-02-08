---
layout: model
title: Word Embeddings for Arabic (arabic_w2v_cc_300d)
author: John Snow Labs
name: arabic_w2v_cc_300d
date: 2020-12-05
task: Embeddings
language: ar
edition: Spark NLP 2.7.0
tags: [embeddings, ar, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained on Common Crawl and Wikipedia using fastText. It is trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.

The model gives 300 dimensional vector outputs per token. The output vectors map words into a meaningful space where the distance between the vectors is related to semantic similarity of words.

These embeddings can be used in multiple tasks like semantic word similarity, named entity recognition, sentiment analysis, and classification.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/arabic_w2v_cc_300d_ar_2.7.0_2.4_1607168354606.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of a pipeline after tokenization.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```python
embeddings = WordEmbeddingsModel.pretrained("arabic_w2v_cc_300d", "ar") \
        .setInputCols(["document", "token"]) \
        .setOutputCol("embeddings")
```
```scala
val embeddings = WordEmbeddingsModel.pretrained("arabic_w2v_cc_300d", "ar") 
        .setInputCols(Array("document", "token"))
        .setOutputCol("embeddings")
```
</div>

{:.h2_title}
## Results
The model gives 300 dimensional Word2Vec feature vector outputs per token.

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|arabic_w2v_cc_300d|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[word_embeddings]|
|Language:|ar|
|Case sensitive:|false|
|Dimension:|300|

## Data Source

This model is imported from [https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html)