---
layout: model
title: Castilian, Spanish luisalberto_pipeline pipeline DistilBertEmbeddings from SantiRimedio
author: John Snow Labs
name: luisalberto_pipeline
date: 2024-09-08
tags: [es, open_source, pipeline, onnx]
task: Embeddings
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`luisalberto_pipeline` is a Castilian, Spanish model originally trained by SantiRimedio.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/luisalberto_pipeline_es_5.5.0_3.0_1725828694841.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/luisalberto_pipeline_es_5.5.0_3.0_1725828694841.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("luisalberto_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("luisalberto_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|luisalberto_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|250.2 MB|

## References

https://huggingface.co/SantiRimedio/LuisAlBERTo

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertEmbeddings