---
layout: model
title: Bert Embeddings Romanian (Base Cased)
author: John Snow Labs
name: bert_base_cased
date: 2021-09-13
tags: [open_source, embeddings, ro]
task: Embeddings
language: ro
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model contains a deep bidirectional transformer trained on Wikipedia and the BookCorpus in Romanian Language. The details are described in the paper “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_cased_ro_3.2.0_3.0_1631533635237.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
embeddings = BertEmbeddings.pretrained("bert_base_cased", "ro") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love NLP']], ["text"]))
```
```scala
...
val embeddings = BertEmbeddings.pretrained("bert_base_cased", "ro")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("I love NLP").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
Generates 768 dimensional embeddings vectors per token
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_cased|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ro|
|Case sensitive:|true|

## Benchmarking

```bash
This model is imported from https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1
```
