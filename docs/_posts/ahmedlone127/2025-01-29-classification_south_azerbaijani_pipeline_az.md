---
layout: model
title: Azerbaijani classification_south_azerbaijani_pipeline pipeline BertForSequenceClassification from language-ml-lab
author: John Snow Labs
name: classification_south_azerbaijani_pipeline
date: 2025-01-29
tags: [az, open_source, pipeline, onnx]
task: Text Classification
language: az
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`classification_south_azerbaijani_pipeline` is a Azerbaijani model originally trained by language-ml-lab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classification_south_azerbaijani_pipeline_az_5.5.1_3.0_1738145840413.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classification_south_azerbaijani_pipeline_az_5.5.1_3.0_1738145840413.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("classification_south_azerbaijani_pipeline", lang = "az")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("classification_south_azerbaijani_pipeline", lang = "az")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classification_south_azerbaijani_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|az|
|Size:|349.7 MB|

## References

https://huggingface.co/language-ml-lab/classification-azb

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification