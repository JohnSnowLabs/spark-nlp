---
layout: model
title: Castilian, Spanish ruperta_base_pipeline pipeline RoBertaEmbeddings from mrm8488
author: John Snow Labs
name: ruperta_base_pipeline
date: 2024-09-19
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ruperta_base_pipeline` is a Castilian, Spanish model originally trained by mrm8488.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ruperta_base_pipeline_es_5.5.0_3.0_1726747293480.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ruperta_base_pipeline_es_5.5.0_3.0_1726747293480.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ruperta_base_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ruperta_base_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ruperta_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|470.0 MB|

## References

https://huggingface.co/mrm8488/RuPERTa-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings