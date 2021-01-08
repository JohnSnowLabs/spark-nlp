---
layout: model
title: Universal Sentence Encoder Multilingual
author: John Snow Labs
name: tfhub_use_multi
date: 2020-12-08
tags: [xx, embeddings, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The Universal Sentence Encoder encodes text into high-dimensional vectors that can be used for text classification, semantic similarity, clustering, and other natural language tasks.

The model is trained and optimized for greater-than-word length text, such as sentences, phrases, or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is the variable-length text and the output is a 512-dimensional vector. The universal-sentence-encoder model has trained with a deep averaging network (DAN) encoder.

This model supports 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian) text encoder.

The details are described in the paper "[Multilingual Universal Sentence Encoder for Semantic Retrieval](https://arxiv.org/abs/1907.04307)".

Note: This model only works on Linux and macOS operating systems and is not compatible with Windows due to the incompatibility of the SentencePiece library.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tfhub_use_multi_xx_2.7.0_2.4_1607427221245.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```python
embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")
```
```scala
val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")
```
</div>

## Results

It gives a 512-dimensional vector of the sentences.

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tfhub_use_multi|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|xx|

## Data Source

This embeddings model is imported from [https://tfhub.dev/google/universal-sentence-encoder-multilingual/3](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)

## Benchmarking

 - We apply this model to the STS benchmark for semantic similarity. The eval can be seen in the [example notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb) made available. Results are shown below:

```bash
STSBenchmark                       | dev    | test  |
-----------------------------------|--------|-------|   
Correlation coefficient of Pearson | 0.829  | 0.809 |
```

 - For semantic similarity retrieval, we evaluate the model on [Quora and AskUbuntu retrieval task.](https://arxiv.org/abs/1811.08008). Results are shown below:

```bash
Dataset                | Quora | AskUbuntu | Average |
-----------------------|-------|-----------|---------|
Mean Averge Precision  | 89.2  | 39.9      | 64.6    |
```

 - For the translation pair retrieval, we evaluate the model on the United Nation Parallal Corpus. Results are shown below:

```bash
Language Pair  | en-es  | en-fr | en-ru | en-zh |
---------------|--------|-------|-------|-------|
Precision@1    | 85.8   | 82.7  | 87.4  | 79.5  |
```