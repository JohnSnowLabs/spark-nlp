---
layout: model
title: English whisper_finetuned_all_3_sumukhshadakshari_pipeline pipeline WhisperForCTC from sumukhshadakshari
author: John Snow Labs
name: whisper_finetuned_all_3_sumukhshadakshari_pipeline
date: 2024-12-17
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_finetuned_all_3_sumukhshadakshari_pipeline` is a English model originally trained by sumukhshadakshari.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_finetuned_all_3_sumukhshadakshari_pipeline_en_5.5.1_3.0_1734401389996.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_finetuned_all_3_sumukhshadakshari_pipeline_en_5.5.1_3.0_1734401389996.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_finetuned_all_3_sumukhshadakshari_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_finetuned_all_3_sumukhshadakshari_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_finetuned_all_3_sumukhshadakshari_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|390.0 MB|

## References

https://huggingface.co/sumukhshadakshari/whisper-finetuned-all-3

## Included Models

- AudioAssembler
- WhisperForCTC