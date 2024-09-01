---
layout: model
title: Castilian, Spanish deberta_embeddings_cowese_base_pipeline pipeline DeBertaEmbeddings from plncmm
author: John Snow Labs
name: deberta_embeddings_cowese_base_pipeline
date: 2024-09-01
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

Pretrained DeBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_embeddings_cowese_base_pipeline` is a Castilian, Spanish model originally trained by plncmm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_embeddings_cowese_base_pipeline_es_5.5.0_3.0_1725230204254.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_embeddings_cowese_base_pipeline_es_5.5.0_3.0_1725230204254.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deberta_embeddings_cowese_base_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deberta_embeddings_cowese_base_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_embeddings_cowese_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|1.0 GB|

## References

https://huggingface.co/plncmm/mdeberta-cowese-base-es

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaEmbeddings