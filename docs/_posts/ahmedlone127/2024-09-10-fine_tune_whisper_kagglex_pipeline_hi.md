---
layout: model
title: Hindi fine_tune_whisper_kagglex_pipeline pipeline WhisperForCTC from SakshiRathi77
author: John Snow Labs
name: fine_tune_whisper_kagglex_pipeline
date: 2024-09-10
tags: [hi, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: hi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fine_tune_whisper_kagglex_pipeline` is a Hindi model originally trained by SakshiRathi77.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fine_tune_whisper_kagglex_pipeline_hi_5.5.0_3.0_1725953862815.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fine_tune_whisper_kagglex_pipeline_hi_5.5.0_3.0_1725953862815.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fine_tune_whisper_kagglex_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fine_tune_whisper_kagglex_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fine_tune_whisper_kagglex_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|1.7 GB|

## References

https://huggingface.co/SakshiRathi77/Fine-tune-Whisper-Kagglex

## Included Models

- AudioAssembler
- WhisperForCTC