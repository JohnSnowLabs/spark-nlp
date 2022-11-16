---
layout: model
title: Word Embeddings for Bengali (bengali_cc_300d)
author: John Snow Labs
name: bengali_cc_300d
date: 2021-02-10
task: Embeddings
language: bn
edition: Spark NLP 2.7.3
spark_version: 2.4
tags: [open_source, bn, embeddings]
supported: true
annotator: WordEmbeddingsModel
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bengali_cc_300d_bn_2.7.3_2.4_1612956925175.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = WordEmbeddingsModel.pretrained("bengali_cc_300d", "bn") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("embeddings")
```



{:.nlu-block}
```python
import nlu
nlu.load("bn.embed").predict("""Put your text here.""")
```

</div>

## Results

```bash
The model gives 300 dimensional feature vector output per token.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bengali_cc_300d|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.7.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[word_embeddings]|
|Language:|bn|
|Case sensitive:|false|
|Dimension:|300|

## Data Source

This model is imported from https://fasttext.cc/docs/en/crawl-vectors.html