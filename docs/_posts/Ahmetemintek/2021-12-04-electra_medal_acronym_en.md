---
layout: model
title: Electra MeDAL Acronym BERT Embeddings
author: Ahmetemintek
name: electra_medal_acronym
date: 2021-12-04
tags: [english, open_source, electra_medal, acronym, embeddings, en]
task: Embeddings
language: en
edition: Spark NLP 3.3.3
spark_version: 3.0
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Electra model fine tuned on MeDAL, a large dataset on abbreviation disambiguation, designed for pretraining natural language understanding models in the medical domain. [reference](https://aclanthology.org/2020.clinicalnlp-1.15.pdf).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/Ahmetemintek/electra_medal_acronym_en_3.3.3_3.0_1638609736515.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler= DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

sentenceDetector = SentenceDetector()\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer= Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = BertEmbeddings.pretrained("electra_medal_ acronym", "en") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")

nlpPipeline= Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, embeddings])
```
```scala
val documentAssembler= DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

val sentenceDetector = SentenceDetector()\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

val tokenizer= Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("electra_medal_ acronym", "en") \
       .setInputCols("sentence", "token") \
       .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_medal_acronym|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[sentence, token]|
|Output Labels:|[electra]|
|Language:|en|
|Case sensitive:|true|

## Data Source

The model is imported from: https://huggingface.co/xhlu/electra-medal