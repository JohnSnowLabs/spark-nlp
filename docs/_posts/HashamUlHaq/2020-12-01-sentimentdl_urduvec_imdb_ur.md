---
layout: model
title: Sentiment Analysis for Urdu (IMDB Review dataset)
author: John Snow Labs
name: sentimentdl_urduvec_imdb
date: 2020-12-01
tags: [sentiment, ur, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Analyse sentiment in reviews by classifying them as ``positive``, ``negative`` or ``neutral``. This model is trained using ``urduvec_140M_300d`` word embeddings. The word embeddings are then converted to sentence embeddings before feeding to the sentiment classifier which uses a DL architecture to classify sentences.

## Predicted Entities

\``positive`` , ``negative`` , ``neutral``

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimentdl_urduvec_imdb_ur_2.7.0_2.4_1606817135630.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, SentenceEmbeddings.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
word_embeddings = WordEmbeddingsModel()\
    .pretrained('urduvec_140M_300d', 'ur')\
    .setInputCols(["document",'token'])\
    .setOutputCol("word_embeddings")

embeddings = SentenceEmbeddings() \
      .setInputCols(["document", "word_embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

classifier = SentimentDLModel.pretrained('sentimentdl_urduvec_imdb', 'ur' )\
    .setInputCols(['document', 'token', 'sentence_embeddings']).setOutputCol('sentiment')

nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, embeddings, sentence_embeddings, classifier])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate(["مجھے واقعی یہ شو سند ہے۔ یہی وجہ ہے کہ مجھے حال ہی میں یہ جان کر مایوسی ہوئی ہے کہ جارج لوپیز ایک ",
                                                                          "بالکل بھی اچھ ،ی کام نہیں کیا گیا ، پوری فلم صرف گرڈج تھی اور کہیں بھی بے ترتیب لوگوں کو ہلاک نہیں"])
```

</div>

## Results

```bash

|    | document                                                                                          | sentiment     |
|---:|:--------------------------------------------------------------------------------------------------|:--------------|
|  0 |مجھے واقعی یہ شو سند ہے۔ یہی وجہ ہے کہ مجھے حال ہی میں یہ جان کر مایوسی ہوئی ہے کہ جارج لوپیز ایک  | positive      |
|  1 |بالکل بھی اچھ ،ی کام نہیں کیا گیا ، پوری فلم صرف گرڈج تھی اور کہیں بھی بے ترتیب لوگوں کو ہلاک نہیں | negative      |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentimentdl_urduvec_imdb|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[sentiment]|
|Language:|ur|
|Dependencies:|urduvec_140M_300d|

## Data Source

This models in trained using data from https://www.kaggle.com/akkefa/imdb-dataset-of-50k-movie-translated-urdu-reviews

## Benchmarking

```bash
loss: 2428.622 - acc: 0.8181 - val_acc: 80.0
```