---
layout: model
title: English hate_speech_detection_mpnet_basev2_pipeline pipeline MPNetForSequenceClassification from Arvnd03
author: John Snow Labs
name: hate_speech_detection_mpnet_basev2_pipeline
date: 2024-09-07
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

Pretrained MPNetForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hate_speech_detection_mpnet_basev2_pipeline` is a English model originally trained by Arvnd03.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hate_speech_detection_mpnet_basev2_pipeline_en_5.5.0_3.0_1725733402271.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hate_speech_detection_mpnet_basev2_pipeline_en_5.5.0_3.0_1725733402271.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hate_speech_detection_mpnet_basev2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hate_speech_detection_mpnet_basev2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hate_speech_detection_mpnet_basev2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.2 MB|

## References

https://huggingface.co/Arvnd03/Hate-Speech-Detection-mpnet-basev2

## Included Models

- DocumentAssembler
- TokenizerModel
- MPNetForSequenceClassification