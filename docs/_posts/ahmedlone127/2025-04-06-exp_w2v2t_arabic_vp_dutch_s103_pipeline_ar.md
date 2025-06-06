---
layout: model
title: Arabic exp_w2v2t_arabic_vp_dutch_s103_pipeline pipeline Wav2Vec2ForCTC from jonatasgrosman
author: John Snow Labs
name: exp_w2v2t_arabic_vp_dutch_s103_pipeline
date: 2025-04-06
tags: [ar, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ar
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`exp_w2v2t_arabic_vp_dutch_s103_pipeline` is a Arabic model originally trained by jonatasgrosman.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/exp_w2v2t_arabic_vp_dutch_s103_pipeline_ar_5.5.1_3.0_1743960337888.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/exp_w2v2t_arabic_vp_dutch_s103_pipeline_ar_5.5.1_3.0_1743960337888.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("exp_w2v2t_arabic_vp_dutch_s103_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("exp_w2v2t_arabic_vp_dutch_s103_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|exp_w2v2t_arabic_vp_dutch_s103_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|1.2 GB|

## References

https://huggingface.co/jonatasgrosman/exp_w2v2t_ar_vp-nl_s103

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC