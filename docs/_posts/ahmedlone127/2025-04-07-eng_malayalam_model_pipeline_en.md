---
layout: model
title: English eng_malayalam_model_pipeline pipeline MarianTransformer from SnehaJay7
author: John Snow Labs
name: eng_malayalam_model_pipeline
date: 2025-04-07
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`eng_malayalam_model_pipeline` is a English model originally trained by SnehaJay7.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/eng_malayalam_model_pipeline_en_5.5.1_3.0_1744019003451.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/eng_malayalam_model_pipeline_en_5.5.1_3.0_1744019003451.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("eng_malayalam_model_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("eng_malayalam_model_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|eng_malayalam_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|262.6 MB|

## References

https://huggingface.co/SnehaJay7/eng-ml-model

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer