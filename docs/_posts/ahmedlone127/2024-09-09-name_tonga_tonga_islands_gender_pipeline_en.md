---
layout: model
title: English name_tonga_tonga_islands_gender_pipeline pipeline BertForSequenceClassification from datalearningpr
author: John Snow Labs
name: name_tonga_tonga_islands_gender_pipeline
date: 2024-09-09
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`name_tonga_tonga_islands_gender_pipeline` is a English model originally trained by datalearningpr.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/name_tonga_tonga_islands_gender_pipeline_en_5.5.0_3.0_1725852716096.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/name_tonga_tonga_islands_gender_pipeline_en_5.5.0_3.0_1725852716096.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("name_tonga_tonga_islands_gender_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("name_tonga_tonga_islands_gender_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|name_tonga_tonga_islands_gender_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|383.3 MB|

## References

https://huggingface.co/datalearningpr/name_to_gender

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification