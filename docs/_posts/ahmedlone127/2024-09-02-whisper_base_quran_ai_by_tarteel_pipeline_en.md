---
layout: model
title: English whisper_base_quran_ai_by_tarteel_pipeline pipeline WhisperForCTC from Democtic
author: John Snow Labs
name: whisper_base_quran_ai_by_tarteel_pipeline
date: 2024-09-02
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_base_quran_ai_by_tarteel_pipeline` is a English model originally trained by Democtic.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_base_quran_ai_by_tarteel_pipeline_en_5.5.0_3.0_1725289788539.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_base_quran_ai_by_tarteel_pipeline_en_5.5.0_3.0_1725289788539.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_base_quran_ai_by_tarteel_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_base_quran_ai_by_tarteel_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_base_quran_ai_by_tarteel_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|643.1 MB|

## References

https://huggingface.co/Democtic/whisper-base-quran-ai-by-tarteel

## Included Models

- AudioAssembler
- WhisperForCTC