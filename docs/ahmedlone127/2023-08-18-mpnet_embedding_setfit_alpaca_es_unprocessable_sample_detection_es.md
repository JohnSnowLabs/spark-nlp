---
layout: model
title: Castilian, Spanish mpnet_embedding_setfit_alpaca_es_unprocessable_sample_detection TFMPNetModel from hackathon-somos-nlp-2023
author: John Snow Labs
name: mpnet_embedding_setfit_alpaca_es_unprocessable_sample_detection
date: 2023-08-18
tags: [mpnet, es, open_source, tensorflow]
task: Embeddings
language: es
edition: Spark NLP 5.1.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MPNetEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained mpnet  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mpnet_embedding_setfit_alpaca_es_unprocessable_sample_detection` is a Castilian, Spanish model originally trained by hackathon-somos-nlp-2023.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mpnet_embedding_setfit_alpaca_es_unprocessable_sample_detection_es_5.1.0_3.0_1692376401533.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mpnet_embedding_setfit_alpaca_es_unprocessable_sample_detection_es_5.1.0_3.0_1692376401533.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

instruction = MPNetEmbeddings \
    .pretrained("mpnet_embedding_setfit_alpaca_es_unprocessable_sample_detection", "es")\
    .setInputCols(["documents"]) \
    .setOutputCol("mpnet_embeddings")

pipeline = Pipeline(stages=[
  document_assembler,
  instruction,
])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)
```
```scala

val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("documents")

val instruction = MPNetEmbeddings
    .pretrained("mpnet_embedding_setfit_alpaca_es_unprocessable_sample_detection", "es")
    .setInputCols(Array("documents")) 
    .setOutputCol("mpnet_embeddings") 

val pipeline = new Pipeline().setStages(Array(document_assembler, instruction))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mpnet_embedding_setfit_alpaca_es_unprocessable_sample_detection|
|Compatibility:|Spark NLP 5.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[mpnet_embeddings]|
|Language:|es|
|Size:|410.2 MB|