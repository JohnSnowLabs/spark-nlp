---
layout: model
title: Castilian, Spanish partypress_monolingual_spain_pipeline pipeline BertForSequenceClassification from partypress
author: John Snow Labs
name: partypress_monolingual_spain_pipeline
date: 2024-09-26
tags: [es, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`partypress_monolingual_spain_pipeline` is a Castilian, Spanish model originally trained by partypress.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/partypress_monolingual_spain_pipeline_es_5.5.0_3.0_1727385906225.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/partypress_monolingual_spain_pipeline_es_5.5.0_3.0_1727385906225.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("partypress_monolingual_spain_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("partypress_monolingual_spain_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|partypress_monolingual_spain_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|412.0 MB|

## References

https://huggingface.co/partypress/partypress-monolingual-spain

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification