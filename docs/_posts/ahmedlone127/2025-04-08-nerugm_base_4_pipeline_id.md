---
layout: model
title: Indonesian nerugm_base_4_pipeline pipeline BertForTokenClassification from apwic
author: John Snow Labs
name: nerugm_base_4_pipeline
date: 2025-04-08
tags: [id, open_source, pipeline, onnx]
task: Named Entity Recognition
language: id
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nerugm_base_4_pipeline` is a Indonesian model originally trained by apwic.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerugm_base_4_pipeline_id_5.5.1_3.0_1744140943621.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nerugm_base_4_pipeline_id_5.5.1_3.0_1744140943621.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nerugm_base_4_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nerugm_base_4_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerugm_base_4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|411.8 MB|

## References

https://huggingface.co/apwic/nerugm-base-4

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification