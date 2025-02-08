---
layout: model
title: Castilian, Spanish oracle_dermat_pipeline pipeline RoBertaForSequenceClassification from fundacionctic
author: John Snow Labs
name: oracle_dermat_pipeline
date: 2025-02-07
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`oracle_dermat_pipeline` is a Castilian, Spanish model originally trained by fundacionctic.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/oracle_dermat_pipeline_es_5.5.1_3.0_1738896700407.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/oracle_dermat_pipeline_es_5.5.1_3.0_1738896700407.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("oracle_dermat_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("oracle_dermat_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|oracle_dermat_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|431.5 MB|

## References

https://huggingface.co/fundacionctic/oracle-dermat

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification