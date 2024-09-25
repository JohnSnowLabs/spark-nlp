---
layout: model
title: Swedish swedish_fine_tuned_whisper_model_nadiaholmlund_pipeline pipeline WhisperForCTC from NadiaHolmlund
author: John Snow Labs
name: swedish_fine_tuned_whisper_model_nadiaholmlund_pipeline
date: 2024-09-17
tags: [sv, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: sv
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`swedish_fine_tuned_whisper_model_nadiaholmlund_pipeline` is a Swedish model originally trained by NadiaHolmlund.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/swedish_fine_tuned_whisper_model_nadiaholmlund_pipeline_sv_5.5.0_3.0_1726563762798.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/swedish_fine_tuned_whisper_model_nadiaholmlund_pipeline_sv_5.5.0_3.0_1726563762798.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("swedish_fine_tuned_whisper_model_nadiaholmlund_pipeline", lang = "sv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("swedish_fine_tuned_whisper_model_nadiaholmlund_pipeline", lang = "sv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|swedish_fine_tuned_whisper_model_nadiaholmlund_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sv|
|Size:|390.9 MB|

## References

https://huggingface.co/NadiaHolmlund/Swedish_Fine_Tuned_Whisper_Model

## Included Models

- AudioAssembler
- WhisperForCTC