---
layout: model
title: Urdu whisper_tiny_urdu_sharjeel103_pipeline pipeline WhisperForCTC from sharjeel103
author: John Snow Labs
name: whisper_tiny_urdu_sharjeel103_pipeline
date: 2024-09-03
tags: [ur, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ur
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_urdu_sharjeel103_pipeline` is a Urdu model originally trained by sharjeel103.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_urdu_sharjeel103_pipeline_ur_5.5.0_3.0_1725365482610.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_urdu_sharjeel103_pipeline_ur_5.5.0_3.0_1725365482610.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_urdu_sharjeel103_pipeline", lang = "ur")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_urdu_sharjeel103_pipeline", lang = "ur")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_urdu_sharjeel103_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ur|
|Size:|389.1 MB|

## References

https://huggingface.co/sharjeel103/whisper-tiny-urdu

## Included Models

- AudioAssembler
- WhisperForCTC