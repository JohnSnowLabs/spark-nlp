---
layout: model
title: English pregnancy_mpnet_embeddings_pipeline pipeline MPNetEmbeddings from AkshaySandbox
author: John Snow Labs
name: pregnancy_mpnet_embeddings_pipeline
date: 2025-03-27
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pregnancy_mpnet_embeddings_pipeline` is a English model originally trained by AkshaySandbox.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pregnancy_mpnet_embeddings_pipeline_en_5.5.1_3.0_1743117538303.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pregnancy_mpnet_embeddings_pipeline_en_5.5.1_3.0_1743117538303.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("pregnancy_mpnet_embeddings_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("pregnancy_mpnet_embeddings_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pregnancy_mpnet_embeddings_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.8 MB|

## References

https://huggingface.co/AkshaySandbox/pregnancy-mpnet-embeddings

## Included Models

- DocumentAssembler
- MPNetEmbeddings