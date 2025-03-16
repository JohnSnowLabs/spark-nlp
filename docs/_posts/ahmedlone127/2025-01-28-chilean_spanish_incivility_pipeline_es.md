---
layout: model
title: Castilian, Spanish chilean_spanish_incivility_pipeline pipeline BertForSequenceClassification from jorgeortizfuentes
author: John Snow Labs
name: chilean_spanish_incivility_pipeline
date: 2025-01-28
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`chilean_spanish_incivility_pipeline` is a Castilian, Spanish model originally trained by jorgeortizfuentes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/chilean_spanish_incivility_pipeline_es_5.5.1_3.0_1738052017276.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/chilean_spanish_incivility_pipeline_es_5.5.1_3.0_1738052017276.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("chilean_spanish_incivility_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("chilean_spanish_incivility_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chilean_spanish_incivility_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|411.6 MB|

## References

https://huggingface.co/jorgeortizfuentes/chilean-spanish-incivility

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification