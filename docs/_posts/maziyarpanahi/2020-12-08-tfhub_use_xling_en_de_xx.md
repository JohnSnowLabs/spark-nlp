---
layout: model
title: Universal Sentence Encoder XLING English and German
author: John Snow Labs
name: tfhub_use_xling_en_de
date: 2020-12-08
tags: [embeddings, open_source, xx]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The Universal Sentence Encoder Cross-lingual (XLING) module is an extension of the Universal Sentence Encoder that includes training on multiple tasks across languages. The multi-task training setup is based on the paper "Learning Cross-lingual Sentence Representations via a Multi-task Dual Encoder".

This specific module is trained on English and German (en-de) tasks, and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and tasks, with the goal of learning text representations that are useful out-of-the-box for a number of applications. The input to the module is variable length English or German text and the output is a 512 dimensional vector. We note that one does not need to specify the language that the input is in, as the model was trained such that English and German text with similar meanings will have similar (high dot product score) embeddings. We also note that this model can be used for monolingual English (and potentially monolingual German) tasks with comparable or even better performance than the purely English Universal Sentence Encoder.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tfhub_use_xling_en_de_xx_2.7.0_2.4_1607440247381.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_xling_en_de", "xx") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")
```
```scala
val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_xling_en_de", "xx")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tfhub_use_xling_en_de|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|xx|

## Data Source

https://tfhub.dev/google/universal-sentence-encoder-xling/en-de/1