---
layout: model
title: English mpnet_embedding_CPU_Economywide_Classifier TFMPNetModel from mtyrrell
author: John Snow Labs
name: mpnet_embedding_CPU_Economywide_Classifier
date: 2023-08-18
tags: [mpnet, en, open_source, tensorflow]
task: Embeddings
language: en
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

Pretrained mpnet  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mpnet_embedding_CPU_Economywide_Classifier` is a English model originally trained by mtyrrell.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mpnet_embedding_CPU_Economywide_Classifier_en_5.1.0_3.0_1692379293006.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mpnet_embedding_CPU_Economywide_Classifier_en_5.1.0_3.0_1692379293006.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

instruction = MPNetEmbeddings \
    .pretrained("mpnet_embedding_CPU_Economywide_Classifier", "en")\
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
    .pretrained("mpnet_embedding_CPU_Economywide_Classifier", "en")
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
|Model Name:|mpnet_embedding_CPU_Economywide_Classifier|
|Compatibility:|Spark NLP 5.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[mpnet_embeddings]|
|Language:|en|
|Size:|410.2 MB|