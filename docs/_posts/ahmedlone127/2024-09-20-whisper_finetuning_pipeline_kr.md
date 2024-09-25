---
layout: model
title: Kanuri whisper_finetuning_pipeline pipeline WhisperForCTC from doongsae
author: John Snow Labs
name: whisper_finetuning_pipeline
date: 2024-09-20
tags: [kr, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: kr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_finetuning_pipeline` is a Kanuri model originally trained by doongsae.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_finetuning_pipeline_kr_5.5.0_3.0_1726876793877.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_finetuning_pipeline_kr_5.5.0_3.0_1726876793877.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_finetuning_pipeline", lang = "kr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_finetuning_pipeline", lang = "kr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_finetuning_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|kr|
|Size:|642.3 MB|

## References

https://huggingface.co/doongsae/whisper_finetuning

## Included Models

- AudioAssembler
- WhisperForCTC