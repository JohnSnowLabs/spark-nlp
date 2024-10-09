---
layout: model
title: Hungarian whisper_tiny_hungarian_475_pipeline pipeline WhisperForCTC from sarpba
author: John Snow Labs
name: whisper_tiny_hungarian_475_pipeline
date: 2024-10-09
tags: [hu, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: hu
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_hungarian_475_pipeline` is a Hungarian model originally trained by sarpba.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_hungarian_475_pipeline_hu_5.5.1_3.0_1728433095913.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_hungarian_475_pipeline_hu_5.5.1_3.0_1728433095913.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_hungarian_475_pipeline", lang = "hu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_hungarian_475_pipeline", lang = "hu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_hungarian_475_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|hu|
|Size:|389.0 MB|

## References

https://huggingface.co/sarpba/whisper-tiny-hungarian_475

## Included Models

- AudioAssembler
- WhisperForCTC