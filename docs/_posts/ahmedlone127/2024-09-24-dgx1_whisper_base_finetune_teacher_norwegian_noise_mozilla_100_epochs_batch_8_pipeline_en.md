---
layout: model
title: English dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_pipeline pipeline WhisperForCTC from rohitp1
author: John Snow Labs
name: dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_pipeline
date: 2024-09-24
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_pipeline` is a English model originally trained by rohitp1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_pipeline_en_5.5.0_3.0_1727146625621.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_pipeline_en_5.5.0_3.0_1727146625621.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|646.9 MB|

## References

https://huggingface.co/rohitp1/dgx1_whisper_base_finetune_teacher_no_noise_mozilla_100_epochs_batch_8

## Included Models

- AudioAssembler
- WhisperForCTC