---
layout: model
title: Universal Sentence Encoder XLING Many
author: John Snow Labs
name: tfhub_use_xling_many
date: 2020-12-08
tags: [embeddings, open_source, xx]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The Universal Sentence Encoder Cross-lingual (XLING) module is an extension of the Universal Sentence Encoder that includes training on multiple tasks across languages. The multi-task training setup is based on the paper "Learning Cross-lingual Sentence Representations via a Multi-task Dual Encoder".

This specific module is trained on English, French, German, Spanish, Italian, Chinese, Korean, and Japanese tasks, and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and tasks, with the goal of learning text representations that are useful out-of-the-box for a number of applications. The input to the module is variable length text in any of the eight aforementioned languages and the output is a 512 dimensional vector. We note that one does not need to specify the language of the input, as the model was trained such that text across languages with similar meanings will have embeddings with high dot product scores.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tfhub_use_xling_many_xx_2.7.0_2.4_1607440840968.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_xling_many", "xx") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")
```
```scala
val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_xling_many", "xx")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tfhub_use_xling_many|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|xx|

## Data Source

https://tfhub.dev/google/universal-sentence-encoder-xling-many/1