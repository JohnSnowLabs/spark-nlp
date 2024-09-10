---
layout: model
title: English asr_nepal_bhasa_yoruba_dummy_pipeline pipeline WhisperForCTC from babs
author: John Snow Labs
name: asr_nepal_bhasa_yoruba_dummy_pipeline
date: 2024-09-10
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_nepal_bhasa_yoruba_dummy_pipeline` is a English model originally trained by babs.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_nepal_bhasa_yoruba_dummy_pipeline_en_5.5.0_3.0_1725940804865.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_nepal_bhasa_yoruba_dummy_pipeline_en_5.5.0_3.0_1725940804865.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("asr_nepal_bhasa_yoruba_dummy_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("asr_nepal_bhasa_yoruba_dummy_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_nepal_bhasa_yoruba_dummy_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|863.9 MB|

## References

https://huggingface.co/babs/ASR-new-yo-dummy

## Included Models

- AudioAssembler
- WhisperForCTC