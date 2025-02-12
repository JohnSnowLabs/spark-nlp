---
layout: model
title: all-MiniLM-L6-v2 converted for SparkNLP
author: glaisney
name: all_MiniLM_L6_v2
date: 2025-02-12
tags: [en, open_source, tensorflow]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.2
supported: false
engine: tensorflow
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

See https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/glaisney/all_MiniLM_L6_v2_en_5.5.1_3.2_1739351970558.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/glaisney/all_MiniLM_L6_v2_en_5.5.1_3.2_1739351970558.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
def embedSentences(sentences: DataFrame): DataFrame = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("sentence")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

        val embeddings = BertEmbeddings
          .pretrained("all_minilm_l6", "en")
          .setInputCols(Array("document", "token"))
          .setOutputCol("embeddings")


    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pipeline = new Pipeline().setStages(Array(
      documentAssembler,
      tokenizer,
      embeddings,
      sentenceEmbeddings
    ))

    val pipelineModel = pipeline.fit(sentences)
    val pipelineDF = pipelineModel.transform(sentences)
```
```scala
def embedSentences(sentences: DataFrame): DataFrame = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("sentence")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

        val embeddings = BertEmbeddings
          .pretrained("all_minilm_l6", "en")
          .setInputCols(Array("document", "token"))
          .setOutputCol("embeddings")


    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pipeline = new Pipeline().setStages(Array(
      documentAssembler,
      tokenizer,
      embeddings,
      sentenceEmbeddings
    ))

    val pipelineModel = pipeline.fit(sentences)
    val pipelineDF = pipelineModel.transform(sentences)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|all_MiniLM_L6_v2|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|84.6 MB|
|Case sensitive:|false|