---
layout: model
title: Russian rubert_address_elements_pipeline pipeline BertForTokenClassification from qwazer
author: John Snow Labs
name: rubert_address_elements_pipeline
date: 2024-11-10
tags: [ru, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rubert_address_elements_pipeline` is a Russian model originally trained by qwazer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rubert_address_elements_pipeline_ru_5.5.1_3.0_1731279362868.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rubert_address_elements_pipeline_ru_5.5.1_3.0_1731279362868.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("rubert_address_elements_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("rubert_address_elements_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rubert_address_elements_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|109.2 MB|

## References

https://huggingface.co/qwazer/rubert-address-elements

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification