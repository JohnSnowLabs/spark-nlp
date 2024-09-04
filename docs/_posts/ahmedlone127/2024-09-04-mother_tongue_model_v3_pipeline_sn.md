---
layout: model
title: Shona mother_tongue_model_v3_pipeline pipeline WhisperForCTC from MothersTongue
author: John Snow Labs
name: mother_tongue_model_v3_pipeline
date: 2024-09-04
tags: [sn, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: sn
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mother_tongue_model_v3_pipeline` is a Shona model originally trained by MothersTongue.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mother_tongue_model_v3_pipeline_sn_5.5.0_3.0_1725430208767.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mother_tongue_model_v3_pipeline_sn_5.5.0_3.0_1725430208767.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mother_tongue_model_v3_pipeline", lang = "sn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mother_tongue_model_v3_pipeline", lang = "sn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mother_tongue_model_v3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sn|
|Size:|1.7 GB|

## References

https://huggingface.co/MothersTongue/mother_tongue_model_v3

## Included Models

- AudioAssembler
- WhisperForCTC