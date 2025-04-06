---
layout: model
title: English modelo_clasificacion_violencia3_pipeline pipeline BertForSequenceClassification from MelisaO
author: John Snow Labs
name: modelo_clasificacion_violencia3_pipeline
date: 2025-04-05
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`modelo_clasificacion_violencia3_pipeline` is a English model originally trained by MelisaO.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/modelo_clasificacion_violencia3_pipeline_en_5.5.1_3.0_1743869884348.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/modelo_clasificacion_violencia3_pipeline_en_5.5.1_3.0_1743869884348.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("modelo_clasificacion_violencia3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("modelo_clasificacion_violencia3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|modelo_clasificacion_violencia3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|667.3 MB|

## References

https://huggingface.co/MelisaO/modelo_clasificacion_violencia3

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification