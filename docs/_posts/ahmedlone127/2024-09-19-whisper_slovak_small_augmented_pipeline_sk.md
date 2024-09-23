---
layout: model
title: Slovak whisper_slovak_small_augmented_pipeline pipeline WhisperForCTC from ALM
author: John Snow Labs
name: whisper_slovak_small_augmented_pipeline
date: 2024-09-19
tags: [sk, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: sk
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_slovak_small_augmented_pipeline` is a Slovak model originally trained by ALM.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_slovak_small_augmented_pipeline_sk_5.5.0_3.0_1726787879617.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_slovak_small_augmented_pipeline_sk_5.5.0_3.0_1726787879617.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_slovak_small_augmented_pipeline", lang = "sk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_slovak_small_augmented_pipeline", lang = "sk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_slovak_small_augmented_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sk|
|Size:|1.7 GB|

## References

https://huggingface.co/ALM/whisper-sk-small-augmented

## Included Models

- AudioAssembler
- WhisperForCTC