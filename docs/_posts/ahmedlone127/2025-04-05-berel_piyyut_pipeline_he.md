---
layout: model
title: Hebrew berel_piyyut_pipeline pipeline BertForTokenClassification from tokeron
author: John Snow Labs
name: berel_piyyut_pipeline
date: 2025-04-05
tags: [he, open_source, pipeline, onnx]
task: Named Entity Recognition
language: he
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`berel_piyyut_pipeline` is a Hebrew model originally trained by tokeron.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/berel_piyyut_pipeline_he_5.5.1_3.0_1743865100767.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/berel_piyyut_pipeline_he_5.5.1_3.0_1743865100767.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("berel_piyyut_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("berel_piyyut_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|berel_piyyut_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|690.1 MB|

## References

https://huggingface.co/tokeron/BEREL_Piyyut

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification