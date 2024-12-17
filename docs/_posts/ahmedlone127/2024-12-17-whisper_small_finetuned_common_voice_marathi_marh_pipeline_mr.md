---
layout: model
title: Marathi whisper_small_finetuned_common_voice_marathi_marh_pipeline pipeline WhisperForCTC from VinayHajare
author: John Snow Labs
name: whisper_small_finetuned_common_voice_marathi_marh_pipeline
date: 2024-12-17
tags: [mr, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: mr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_finetuned_common_voice_marathi_marh_pipeline` is a Marathi model originally trained by VinayHajare.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_finetuned_common_voice_marathi_marh_pipeline_mr_5.5.1_3.0_1734402381016.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_finetuned_common_voice_marathi_marh_pipeline_mr_5.5.1_3.0_1734402381016.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("whisper_small_finetuned_common_voice_marathi_marh_pipeline", lang = "mr")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("whisper_small_finetuned_common_voice_marathi_marh_pipeline", lang = "mr")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_finetuned_common_voice_marathi_marh_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|mr|
|Size:|1.7 GB|

## References

References

https://huggingface.co/VinayHajare/whisper-small-finetuned-common-voice-mr

## Included Models

- AudioAssembler
- WhisperForCTC