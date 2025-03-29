---
layout: model
title: Indonesian dkbli_pipeline pipeline BertForSequenceClassification from ketut
author: John Snow Labs
name: dkbli_pipeline
date: 2025-03-28
tags: [id, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dkbli_pipeline` is a Indonesian model originally trained by ketut.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dkbli_pipeline_id_5.5.1_3.0_1743159744049.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dkbli_pipeline_id_5.5.1_3.0_1743159744049.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dkbli_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dkbli_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dkbli_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|469.0 MB|

## References

https://huggingface.co/ketut/dKBLI

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification