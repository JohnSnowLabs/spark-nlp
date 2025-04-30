---
layout: model
title: Persian named_entity_recognition_persian_pipeline pipeline BertForTokenClassification from NLPclass
author: John Snow Labs
name: named_entity_recognition_persian_pipeline
date: 2025-03-29
tags: [fa, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fa
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`named_entity_recognition_persian_pipeline` is a Persian model originally trained by NLPclass.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/named_entity_recognition_persian_pipeline_fa_5.5.1_3.0_1743258593751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/named_entity_recognition_persian_pipeline_fa_5.5.1_3.0_1743258593751.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("named_entity_recognition_persian_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("named_entity_recognition_persian_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|named_entity_recognition_persian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|606.6 MB|

## References

https://huggingface.co/NLPclass/Named_entity_recognition_persian

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification