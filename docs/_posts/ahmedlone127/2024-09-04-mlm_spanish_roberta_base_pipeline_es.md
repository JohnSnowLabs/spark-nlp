---
layout: model
title: Castilian, Spanish mlm_spanish_roberta_base_pipeline pipeline RoBertaEmbeddings from MMG
author: John Snow Labs
name: mlm_spanish_roberta_base_pipeline
date: 2024-09-04
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mlm_spanish_roberta_base_pipeline` is a Castilian, Spanish model originally trained by MMG.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mlm_spanish_roberta_base_pipeline_es_5.5.0_3.0_1725412211265.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mlm_spanish_roberta_base_pipeline_es_5.5.0_3.0_1725412211265.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mlm_spanish_roberta_base_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mlm_spanish_roberta_base_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mlm_spanish_roberta_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|470.9 MB|

## References

https://huggingface.co/MMG/mlm-spanish-roberta-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings