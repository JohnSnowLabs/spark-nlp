---
layout: model
title: Dutch, Flemish autotrain_nes_dutch_63520135542_pipeline pipeline BertForTokenClassification from peanutacake
author: John Snow Labs
name: autotrain_nes_dutch_63520135542_pipeline
date: 2025-02-08
tags: [nl, open_source, pipeline, onnx]
task: Named Entity Recognition
language: nl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_nes_dutch_63520135542_pipeline` is a Dutch, Flemish model originally trained by peanutacake.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_nes_dutch_63520135542_pipeline_nl_5.5.1_3.0_1738985684584.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_nes_dutch_63520135542_pipeline_nl_5.5.1_3.0_1738985684584.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_nes_dutch_63520135542_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_nes_dutch_63520135542_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_nes_dutch_63520135542_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|406.8 MB|

## References

https://huggingface.co/peanutacake/autotrain-nes_nl-63520135542

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification