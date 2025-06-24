---
layout: model
title: Italian feel_italian_italian_emotion_pipeline pipeline CamemBertForSequenceClassification from MilaNLProc
author: John Snow Labs
name: feel_italian_italian_emotion_pipeline
date: 2025-06-22
tags: [it, open_source, pipeline, onnx]
task: Text Classification
language: it
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`feel_italian_italian_emotion_pipeline` is a Italian model originally trained by MilaNLProc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/feel_italian_italian_emotion_pipeline_it_5.5.1_3.0_1750620662830.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/feel_italian_italian_emotion_pipeline_it_5.5.1_3.0_1750620662830.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("feel_italian_italian_emotion_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("feel_italian_italian_emotion_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|feel_italian_italian_emotion_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|394.8 MB|

## References

https://huggingface.co/MilaNLProc/feel-it-italian-emotion

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForSequenceClassification