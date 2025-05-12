---
layout: model
title: English w2v2_base_pretrained_lr5e_5_at0_1_da1_pipeline pipeline Wav2Vec2ForCTC from MelanieKoe
author: John Snow Labs
name: w2v2_base_pretrained_lr5e_5_at0_1_da1_pipeline
date: 2025-04-01
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

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`w2v2_base_pretrained_lr5e_5_at0_1_da1_pipeline` is a English model originally trained by MelanieKoe.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/w2v2_base_pretrained_lr5e_5_at0_1_da1_pipeline_en_5.5.1_3.0_1743543692785.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/w2v2_base_pretrained_lr5e_5_at0_1_da1_pipeline_en_5.5.1_3.0_1743543692785.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("w2v2_base_pretrained_lr5e_5_at0_1_da1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("w2v2_base_pretrained_lr5e_5_at0_1_da1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|w2v2_base_pretrained_lr5e_5_at0_1_da1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|348.7 MB|

## References

https://huggingface.co/MelanieKoe/w2v2-base-pretrained_lr5e-5_at0.1_da1

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC