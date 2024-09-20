---
layout: model
title: Guarani whisper_base_wer_pipeline pipeline WhisperForCTC from adrianSauer
author: John Snow Labs
name: whisper_base_wer_pipeline
date: 2024-09-10
tags: [gn, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: gn
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_base_wer_pipeline` is a Guarani model originally trained by adrianSauer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_base_wer_pipeline_gn_5.5.0_3.0_1725953648617.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_base_wer_pipeline_gn_5.5.0_3.0_1725953648617.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_base_wer_pipeline", lang = "gn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_base_wer_pipeline", lang = "gn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_base_wer_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|gn|
|Size:|642.3 MB|

## References

https://huggingface.co/adrianSauer/whisper-base-wer

## Included Models

- AudioAssembler
- WhisperForCTC