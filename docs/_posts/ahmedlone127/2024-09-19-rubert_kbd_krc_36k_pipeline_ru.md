---
layout: model
title: Russian rubert_kbd_krc_36k_pipeline pipeline BertForSequenceClassification from alimboff
author: John Snow Labs
name: rubert_kbd_krc_36k_pipeline
date: 2024-09-19
tags: [ru, open_source, pipeline, onnx]
task: Text Classification
language: ru
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rubert_kbd_krc_36k_pipeline` is a Russian model originally trained by alimboff.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rubert_kbd_krc_36k_pipeline_ru_5.5.0_3.0_1726707119563.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rubert_kbd_krc_36k_pipeline_ru_5.5.0_3.0_1726707119563.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("rubert_kbd_krc_36k_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("rubert_kbd_krc_36k_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rubert_kbd_krc_36k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|44.2 MB|

## References

https://huggingface.co/alimboff/rubert-kbd-krc-36K

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification