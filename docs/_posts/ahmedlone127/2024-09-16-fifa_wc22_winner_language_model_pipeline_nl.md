---
layout: model
title: Dutch, Flemish fifa_wc22_winner_language_model_pipeline pipeline WhisperForCTC from AlexMo
author: John Snow Labs
name: fifa_wc22_winner_language_model_pipeline
date: 2024-09-16
tags: [nl, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: nl
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fifa_wc22_winner_language_model_pipeline` is a Dutch, Flemish model originally trained by AlexMo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fifa_wc22_winner_language_model_pipeline_nl_5.5.0_3.0_1726480299895.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fifa_wc22_winner_language_model_pipeline_nl_5.5.0_3.0_1726480299895.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fifa_wc22_winner_language_model_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fifa_wc22_winner_language_model_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fifa_wc22_winner_language_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|1.7 GB|

## References

https://huggingface.co/AlexMo/FIFA_WC22_WINNER_LANGUAGE_MODEL

## Included Models

- AudioAssembler
- WhisperForCTC