---
layout: model
title: Thai exp_w2v2t_thai_xlsr_53_s201_pipeline pipeline Wav2Vec2ForCTC from jonatasgrosman
author: John Snow Labs
name: exp_w2v2t_thai_xlsr_53_s201_pipeline
date: 2025-03-29
tags: [th, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: th
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`exp_w2v2t_thai_xlsr_53_s201_pipeline` is a Thai model originally trained by jonatasgrosman.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/exp_w2v2t_thai_xlsr_53_s201_pipeline_th_5.5.1_3.0_1743288519869.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/exp_w2v2t_thai_xlsr_53_s201_pipeline_th_5.5.1_3.0_1743288519869.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("exp_w2v2t_thai_xlsr_53_s201_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("exp_w2v2t_thai_xlsr_53_s201_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|exp_w2v2t_thai_xlsr_53_s201_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|1.2 GB|

## References

https://huggingface.co/jonatasgrosman/exp_w2v2t_th_xlsr-53_s201

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC